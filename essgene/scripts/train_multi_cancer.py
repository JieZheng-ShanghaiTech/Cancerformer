"""
Train gene essentiality classifier on merged multi-cancer data.

Corresponds to the original geneformer_ess_merge.py.
Trains sequentially on multiple cancer types using the previous model as base.

Usage:
    python -m essgene.scripts.train_merge --cancer cesc --cell_line caski --only_train
"""

import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from transformers import Trainer
from transformers.training_args import TrainingArguments

from geneformer import DataCollatorForGeneClassification
from geneformer import classifier_utils as cu
from geneformer import perturber_utils as pu

from essgene.data import (
    load_config,
    load_expression_features,
    load_gene_info,
    load_essentiality_labels,
    prepare_feature_dicts,
    build_gene_class_dicts,
)
from essgene.data.loader import load_scgpt_embeddings, load_token_dict
from essgene.evaluation.metrics import cal_corscores
from essgene.geneformer_patches import EssClassifier
from essgene.geneformer_patches.classifier_utils_patch import (
    prep_gene_classifier_all_data_with_scores,
    validate_and_clean_cols_with_scores,
)
from essgene.models import BertForEssGeneClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-cancer merged essentiality classifier")
    parser.add_argument("--cancer", type=str, default="gbm", help="Target cancer for testing")
    parser.add_argument("--cell_line", type=str, default="caski")
    parser.add_argument("--layers", type=int, default=6, choices=[6, 12])
    parser.add_argument("--freeze_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--subsample_size", type=int, default=10_000)
    parser.add_argument("--exclude_pan", action="store_true")
    parser.add_argument("--only_test", action="store_true")
    parser.add_argument("--only_train", action="store_true")
    parser.add_argument("--suf", type=str, default="")
    parser.add_argument("--sample", type=str, default="")
    parser.add_argument("--train_cancers", type=str, nargs="+",
                        default=["cesc", "gbm", "luad"],
                        help="Cancer types to train on sequentially")
    parser.add_argument("--train_cell_lines", type=str, nargs="+",
                        default=["siha", "gbm", "luad"],
                        help="Cell lines for each training cancer")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    gene_token_dict = load_token_dict()
    gene_info_table, gene_info_dict = load_gene_info(config)
    scgpt_embed = load_scgpt_embeddings(config)

    # ===== SEQUENTIAL TRAINING =====
    if not args.only_test:
        base_model = config["models"][f"geneformer_{args.layers}l"]

        for i, (cancer, cell_line) in enumerate(zip(args.train_cancers, args.train_cell_lines)):
            print(f"\n=== Training on {cancer} ({cell_line}) ===")

            exp_feat_dict_raw = load_expression_features(cancer, config)
            ess_label_df = load_essentiality_labels(cancer, config, cell_line)

            unip_feat_dict_raw = {
                gene_info_dict.get(gene): feat
                for gene, feat in scgpt_embed.items()
                if gene in gene_info_dict
            }
            inter_ensembl = (
                set(ess_label_df["ensembl_id"])
                & set(unip_feat_dict_raw.keys())
                & set(gene_token_dict.keys())
            )
            exp_feat_dict, unip_feat_dict = prepare_feature_dicts(
                exp_feat_dict_raw, scgpt_embed, gene_info_dict, gene_token_dict, inter_ensembl
            )

            _, gene_score_dict, train_gene_class_dict, _ = (
                build_gene_class_dicts(ess_label_df, args.exclude_pan)
            )

            dataset_path = config["datasets"][cancer]["train"]
            dataset_path = dataset_path.replace("{cell_line}", cell_line)

            output_base = config["output"].get(cancer, config["output"][args.cancer])
            train_history = "_".join(args.train_cancers[:i + 1])
            suffix = f"train_{args.layers}l_{cell_line}_freez_{args.freeze_layers}_train_on_{train_history}"
            output_dir = f"{output_base}/{suffix}"
            os.makedirs(output_dir, exist_ok=True)

            output_prefix = "ess_pred_with_omics"
            training_args = {
                "evaluation_strategy": "no",
                "save_strategy": "steps",
                "save_steps": 42,
                "num_train_epochs": args.epochs,
            }

            cc = EssClassifier(
                classifier="gene",
                gene_class_dict=train_gene_class_dict,
                gene_score_dict=gene_score_dict,
                max_ncells=args.subsample_size,
                freeze_layers=args.freeze_layers,
                training_args=training_args,
                num_crossval_splits=0,
                forward_batch_size=100,
                nproc=16,
            )

            cc.prepare_data(
                input_data_file=dataset_path,
                output_directory=output_dir,
                output_prefix=output_prefix,
                exp_feat_dict=exp_feat_dict,
                unip_feat_dict=unip_feat_dict,
            )

            cc.train_all_data(
                model_directory=base_model,
                exp_feat_dict=exp_feat_dict,
                unip_feat_dict=unip_feat_dict,
                prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
                id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
                output_directory=output_dir,
                output_prefix=output_prefix,
            )

            # Use this model as base for next cancer
            base_model = f"{output_dir}/geneformer_geneClassifier_{output_prefix}"

    # ===== TESTING =====
    if not args.only_train:
        torch.cuda.empty_cache()
        cancer = args.cancer
        cell_line = args.cell_line

        exp_feat_dict_raw = load_expression_features(cancer, config)
        ess_label_df = load_essentiality_labels(cancer, config, cell_line)

        unip_feat_dict_raw = {
            gene_info_dict.get(gene): feat
            for gene, feat in scgpt_embed.items()
            if gene in gene_info_dict
        }
        inter_ensembl = (
            set(ess_label_df["ensembl_id"])
            & set(unip_feat_dict_raw.keys())
            & set(gene_token_dict.keys())
        )
        exp_feat_dict, unip_feat_dict = prepare_feature_dicts(
            exp_feat_dict_raw, scgpt_embed, gene_info_dict, gene_token_dict, inter_ensembl
        )

        _, gene_score_dict, _, test_gene_class_dict = (
            build_gene_class_dicts(ess_label_df, args.exclude_pan)
        )

        cl = cell_line
        dataset_path_test = config["datasets"][cancer]["test"]
        for placeholder, val in [("{cell_line}", cell_line), ("{sample}", args.sample)]:
            dataset_path_test = dataset_path_test.replace(placeholder, val)

        output_base = config["output"][cancer]
        output_prefix = "ess_pred_with_omics"
        train_history = "_".join(args.train_cancers)
        suffix = f"test_merge_{args.layers}l_{cl}_sample{args.sample}"
        output_dir = f"{output_base}/{suffix}"
        os.makedirs(output_dir, exist_ok=True)

        # Find the last trained model
        last_cancer = args.train_cancers[-1]
        last_cl = args.train_cell_lines[-1]
        finetuned_base = (
            f"{config['output'].get(last_cancer, output_base)}"
            f"/train_{args.layers}l_{last_cl}_freez_{args.freeze_layers}_train_on_{train_history}"
            f"/geneformer_geneClassifier_{output_prefix}"
        )
        checkpoint_list = [f"{finetuned_base}/checkpoint-{42 * p}" for p in range(1, 1000)]

        ct = EssClassifier(
            classifier="gene",
            gene_class_dict=test_gene_class_dict,
            gene_score_dict=gene_score_dict,
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
            exp_feat_dict=exp_feat_dict,
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
        gene_score_dict_tok = {gene_token_dict.get(k): v for k, v in gene_score_dict.items()}
        scores = [gene_score_dict_tok[k] for k in targets]

        data = prep_gene_classifier_all_data_with_scores(
            data, targets, labels, scores, args.subsample_size, 16, exp_feat_dict
        )
        eval_data, _ = validate_and_clean_cols_with_scores(data, None, "gene")

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
                model = BertForEssGeneClassification.from_pretrained(
                    ckp_path, num_labels=2,
                    output_hidden_states=False, output_attentions=False,
                    exp_feat_dict=exp_feat_dict, unip_feat_dict=unip_feat_dict,
                )
            except Exception:
                continue

            model = model.to("cuda")
            trainer = Trainer(
                model=model, args=eval_training_args,
                data_collator=DataCollatorForGeneClassification(),
                eval_dataset=eval_data,
            )

            outputs = trainer.predict(eval_data)
            all_labels = outputs.label_ids.flatten()
            all_logits = outputs.predictions.reshape(-1, 2)

            all_input_ids = []
            for seq in eval_data["input_ids"]:
                if len(seq) < 2048:
                    seq = seq + [0] * (2048 - len(seq))
                all_input_ids.extend(seq)
            all_input_ids = np.asarray(all_input_ids)[:, np.newaxis]

            real_logits = all_logits[all_labels != -100]
            real_genes = list(all_input_ids[all_labels != -100].flatten())

            pred_dict = {
                "gene_n_list": [ensembl_gene_dict[token_gene_dict[i]] for i in real_genes],
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
