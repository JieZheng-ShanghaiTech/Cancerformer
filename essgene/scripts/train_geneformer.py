"""
Train gene essentiality classifier using original geneformer (no omics features).

Corresponds to the original geneformer_ess_ori.py.
Uses standard BertForTokenClassification without expression/protein features.

Usage:
    python -m essgene.scripts.train_original --cancer cesc --only_train
"""

import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from transformers import BertForTokenClassification, Trainer
from transformers.training_args import TrainingArguments

from geneformer import Classifier, DataCollatorForGeneClassification
from geneformer import classifier_utils as cu
from geneformer import perturber_utils as pu

from essgene.data import (
    load_config,
    load_gene_info,
    load_essentiality_labels,
    build_gene_class_dicts,
)
from essgene.data.loader import load_token_dict
from essgene.evaluation.metrics import cal_corscores


def parse_args():
    parser = argparse.ArgumentParser(description="Train original geneformer classifier (no omics)")
    parser.add_argument("--cancer", type=str, default="gbm")
    parser.add_argument("--cell_line", type=str, default="caski")
    parser.add_argument("--layers", type=int, default=6, choices=[6, 12])
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--subsample_size", type=int, default=10_000)
    parser.add_argument("--exclude_pan", action="store_true")
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--only_train", action="store_true")
    parser.add_argument("--suf", type=str, default="")
    parser.add_argument("--sample", type=str, default="")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    gene_token_dict = load_token_dict()
    gene_info_table, gene_info_dict = load_gene_info(config)
    ess_label_df = load_essentiality_labels(args.cancer, config, args.cell_line)

    gene_class_dict, gene_score_dict, train_gene_class_dict, test_gene_class_dict = (
        build_gene_class_dicts(ess_label_df, args.exclude_pan)
    )

    cl = args.cell_line or config["labels"][args.cancer].split("/")[-1].split("_")[0]
    dataset_path_train = config["datasets"][args.cancer]["train"]
    dataset_path_test = config["datasets"][args.cancer]["test"]
    for placeholder, val in [("{cell_line}", args.cell_line), ("{sample}", args.sample)]:
        dataset_path_train = dataset_path_train.replace(placeholder, val)
        dataset_path_test = dataset_path_test.replace(placeholder, val)

    # ===== TRAINING (uses standard Classifier, no omics) =====
    if not args.only_test:
        base_model = config["models"][f"geneformer_{args.layers}l"]
        output_base = config["output"][args.cancer]
        suffix = f"train_ori_{args.layers}l_{cl}_freez_{args.freeze_layers}_{args.suf}"
        output_dir = f"{output_base}/{suffix}"
        os.makedirs(output_dir, exist_ok=True)

        output_prefix = "ess_pred_original"
        training_args = {
            "evaluation_strategy": "no",
            "save_strategy": "steps",
            "save_steps": 42,
            "num_train_epochs": args.epochs,
        }

        cc = Classifier(
            classifier="gene",
            gene_class_dict=train_gene_class_dict,
            max_ncells=args.subsample_size,
            freeze_layers=args.freeze_layers,
            training_args=training_args,
            num_crossval_splits=0,
            forward_batch_size=100,
            nproc=16,
        )

        cc.prepare_data(
            input_data_file=dataset_path_train,
            output_directory=output_dir,
            output_prefix=output_prefix,
        )

        cc.train_all_data(
            model_directory=base_model,
            prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
            id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
            output_directory=output_dir,
            output_prefix=output_prefix,
        )

    # ===== TESTING =====
    if not args.only_train:
        torch.cuda.empty_cache()
        output_base = config["output"][args.cancer]
        output_prefix = "ess_pred_original"

        suffix = f"test_ori_{args.layers}l_{cl}_freez_{args.freeze_layers}_{args.suf}"
        finetuned_base = (
            f"{output_base}/train_ori_{args.layers}l_{cl}_freez_{args.freeze_layers}_{args.suf}"
            f"/geneformer_geneClassifier_{output_prefix}"
        )
        checkpoint_list = [f"{finetuned_base}/checkpoint-{42 * p}" for p in range(1, 200)]

        output_dir = f"{output_base}/{suffix}"
        os.makedirs(output_dir, exist_ok=True)

        ct = Classifier(
            classifier="gene",
            gene_class_dict=test_gene_class_dict,
            max_ncells=args.subsample_size,
            freeze_layers=args.freeze_layers,
            num_crossval_splits=0,
            forward_batch_size=100,
            nproc=16,
        )
        ct.prepare_data(
            input_data_file=dataset_path_test,
            output_directory=output_dir,
            output_prefix=f"{output_prefix}_{cl}",
        )

        with open(f"{output_dir}/{output_prefix}_{cl}_id_class_dict.pkl", "rb") as f:
            id_class_dict = pkl.load(f)
        class_id_dict = {v: k for k, v in id_class_dict.items()}

        data = pu.load_and_filter(None, 16, f"{output_dir}/{output_prefix}_{cl}_labeled.dataset")
        data = data.shuffle(seed=42)

        test_gene_class_dict_tok = {
            k: set([gene_token_dict.get(gene) for gene in v])
            for k, v in test_gene_class_dict.items()
        }
        targets = pu.flatten_list(test_gene_class_dict_tok.values())
        labels = pu.flatten_list([
            [class_id_dict[label]] * len(tgts)
            for label, tgts in test_gene_class_dict_tok.items()
        ])

        token_gene_dict = {v: k for k, v in gene_token_dict.items()}
        ensembl_gene_dict = gene_info_table.set_index("ensembl_id")["gene_name"].to_dict()

        eval_training_args = TrainingArguments(
            num_train_epochs=args.epochs,
            do_train=False, do_eval=True,
            evaluation_strategy="no",
            group_by_length=False, length_column_name="length",
            disable_tqdm=False, output_dir=output_dir,
            per_device_eval_batch_size=100,
        )

        for ckp_path in checkpoint_list:
            ckp = ckp_path.split("/")[-1]
            ckp_dir = f"{output_dir}/{ckp}"
            os.makedirs(ckp_dir, exist_ok=True)

            try:
                model = BertForTokenClassification.from_pretrained(
                    ckp_path, num_labels=2,
                    output_hidden_states=False, output_attentions=False,
                )
            except Exception:
                continue

            model = model.to("cuda")
            trainer = Trainer(
                model=model, args=eval_training_args,
                data_collator=DataCollatorForGeneClassification(),
                eval_dataset=data,
            )

            outputs = trainer.predict(data)
            all_labels = outputs.label_ids.flatten()
            all_logits = outputs.predictions.reshape(-1, 2)

            all_input_ids = []
            for seq in data["input_ids"]:
                if len(seq) < 2048:
                    seq = seq + [0] * (2048 - len(seq))
                all_input_ids.extend(seq)
            all_input_ids = np.asarray(all_input_ids)[:, np.newaxis]

            real_logits = all_logits[all_labels != -100]
            real_genes = list(all_input_ids[all_labels != -100].flatten())

            pred_dict = {
                "gene_n_list": [ensembl_gene_dict.get(token_gene_dict.get(i, ""), "") for i in real_genes],
                "y_score_pos": real_logits[:, 1],
                "y_score_neg": 1 - real_logits[:, 1],
            }
            with open(f"{ckp_dir}/pred_dict.pkl", "wb") as f:
                pkl.dump(pred_dict, f)

            pred_df = pd.DataFrame(pred_dict)
            new_df = (
                pred_df.groupby("gene_n_list")
                .apply(lambda g: g.sort_values("y_score_pos", ascending=False)[["y_score_pos", "y_score_neg"]][:100].mean())
            )
            new_df["gene_n_list"] = new_df.index.values
            new_df.to_csv(f"{ckp_dir}/mean_score.csv")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
