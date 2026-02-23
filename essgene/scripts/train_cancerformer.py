"""
CancerFormer: Gene essentiality classifier with optional PPI-GAT integration.

Supports:
- Single-cancer and multi-cancer training modes
- Ablation studies (disable expression/protein/PPI features)
- Full feature integration (Geneformer + expression + protein + PPI-GAT)

Usage:
    # Single cancer with all features
    python -m essgene.scripts.train_cancerformer --cancer gbm --only_train

    # Single cancer with ablations
    python -m essgene.scripts.train_cancerformer --cancer gbm --no_ppi --only_train
    python -m essgene.scripts.train_cancerformer --cancer gbm --no_exp --no_protein --only_train

    # Multi-cancer training
    python -m essgene.scripts.train_cancerformer --multi_cancer \
        --train_cancers cesc gbm luad --train_cell_lines siha gbm luad \
        --cancer gbm --only_train
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
    load_ppi_graph,
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
    parser = argparse.ArgumentParser(
        description="CancerFormer: Train gene essentiality classifier with optional PPI-GAT"
    )

    # Basic training arguments
    parser.add_argument("--cancer", type=str, default="gbm",
                        help="Cancer type for single-cancer mode or test target for multi-cancer")
    parser.add_argument("--cell_line", type=str, default="",
                        help="Cell line name (e.g., caski, hela)")
    parser.add_argument("--layers", type=int, default=6, choices=[6, 12],
                        help="Geneformer layers (6 or 12)")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of BERT layers to freeze")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs")
    parser.add_argument("--subsample_size", type=int, default=10_000,
                        help="Max cells per gene")

    # Training control
    parser.add_argument("--exclude_pan", action="store_true",
                        help="Exclude pan-cancer essential genes from training")
    parser.add_argument("--only_test", action="store_true",
                        help="Run only testing phase")
    parser.add_argument("--only_train", action="store_true",
                        help="Run only training phase")

    # Multi-cancer mode
    parser.add_argument("--multi_cancer", action="store_true",
                        help="Enable multi-cancer sequential training mode")
    parser.add_argument("--train_cancers", type=str, nargs="+",
                        default=["cesc", "gbm", "luad"],
                        help="Cancer types to train on sequentially (multi-cancer mode)")
    parser.add_argument("--train_cell_lines", type=str, nargs="+",
                        default=["siha", "gbm", "luad"],
                        help="Cell lines for each training cancer (multi-cancer mode)")

    # Ablation studies
    parser.add_argument("--no_exp", action="store_true",
                        help="Ablation: Disable expression features")
    parser.add_argument("--no_protein", action="store_true",
                        help="Ablation: Disable protein/scGPT features")
    parser.add_argument("--no_ppi", action="store_true",
                        help="Ablation: Disable PPI-GAT module")

    # Other
    parser.add_argument("--suf", type=str, default="",
                        help="Suffix for output directory")
    parser.add_argument("--sample", type=str, default="",
                        help="Sample identifier for path resolution")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")

    return parser.parse_args()


def load_features(args, config, cancer, cell_line, gene_token_dict, gene_info_dict, scgpt_embed):
    """Load and prepare features based on ablation flags."""
    # Load expression features
    exp_feat_dict_raw = load_expression_features(cancer, config) if not args.no_exp else {}

    # Load essentiality labels
    ess_label_df = load_essentiality_labels(cancer, config, cell_line)

    # Prepare protein features
    if not args.no_protein:
        unip_feat_dict_raw = {
            gene_info_dict.get(gene): feat
            for gene, feat in scgpt_embed.items()
            if gene in gene_info_dict
        }
    else:
        unip_feat_dict_raw = {}

    # Compute intersection
    inter_ensembl = set(ess_label_df["ensembl_id"]) & set(gene_token_dict.keys())
    if not args.no_exp:
        inter_ensembl &= set(exp_feat_dict_raw.keys())
    if not args.no_protein:
        inter_ensembl &= set(unip_feat_dict_raw.keys())

    # Prepare feature dicts
    exp_feat_dict, unip_feat_dict = prepare_feature_dicts(
        exp_feat_dict_raw, scgpt_embed, gene_info_dict, gene_token_dict, inter_ensembl
    )

    # Apply ablations
    if args.no_exp:
        exp_feat_dict = None
    if args.no_protein:
        unip_feat_dict = None

    print(f"Expression features: {len(exp_feat_dict) if exp_feat_dict else 0}, "
          f"Protein features: {len(unip_feat_dict) if unip_feat_dict else 0}")

    return exp_feat_dict, unip_feat_dict, ess_label_df


def get_suffix(args, cl, train_history=None):
    """Generate output directory suffix based on configuration."""
    ablations = []
    if args.no_exp:
        ablations.append("noexp")
    if args.no_protein:
        ablations.append("noprot")
    if args.no_ppi:
        ablations.append("noppi")

    ablation_str = "_".join(ablations) if ablations else "full"

    if train_history:
        return f"train_{args.layers}l_{cl}_freez_{args.freeze_layers}_{ablation_str}_on_{train_history}_{args.suf}"
    else:
        base = f"train_{args.layers}l_{cl}_freez_{args.freeze_layers}_{ablation_str}"
        if args.exclude_pan:
            base += "_wo_pan"
        return f"{base}_{args.suf}" if args.suf else base


def train_single_cancer(args, config, cancer, cell_line, base_model,
                       gene_token_dict, gene_info_table, gene_info_dict, scgpt_embed):
    """Train on a single cancer type."""
    print(f"\n=== Training on {cancer} ({cell_line}) ===")

    # Load features
    exp_feat_dict, unip_feat_dict, ess_label_df = load_features(
        args, config, cancer, cell_line, gene_token_dict, gene_info_dict, scgpt_embed
    )

    # Load PPI graph if not ablated
    ppi_edge_index = None
    if not args.no_ppi and exp_feat_dict and unip_feat_dict:
        ppi_edge_index = load_ppi_graph(config, gene_info_table, gene_token_dict, exp_feat_dict)
        print(f"PPI edges: {ppi_edge_index.shape[1] if ppi_edge_index is not None else 0}")

    # Build class dicts
    _, gene_score_dict, train_gene_class_dict, _ = build_gene_class_dicts(
        ess_label_df, args.exclude_pan
    )

    # Resolve paths
    cl = cell_line or config["labels"][cancer].split("/")[-1].split("_")[0]
    dataset_path = config["datasets"][cancer]["train"]
    dataset_path = dataset_path.replace("{cell_line}", cell_line).replace("{sample}", args.sample)

    # Setup output directory
    output_base = config["output"][cancer]
    suffix = get_suffix(args, cl)
    output_dir = f"{output_base}/{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    output_prefix = "ess_pred_with_omics"
    training_args = {
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 42,
        "num_train_epochs": args.epochs,
    }

    # Train
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
        ppi_edge_index=ppi_edge_index,
    )

    cc.train_all_data(
        model_directory=base_model,
        exp_feat_dict=exp_feat_dict,
        unip_feat_dict=unip_feat_dict,
        ppi_edge_index=ppi_edge_index,
        prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

    return f"{output_dir}/geneformer_geneClassifier_{output_prefix}"


def train_multi_cancer(args, config, gene_token_dict, gene_info_table, gene_info_dict, scgpt_embed):
    """Train sequentially on multiple cancer types."""
    base_model = config["models"][f"geneformer_{args.layers}l"]

    for i, (cancer, cell_line) in enumerate(zip(args.train_cancers, args.train_cell_lines)):
        print(f"\n=== Training on {cancer} ({cell_line}) [{i+1}/{len(args.train_cancers)}] ===")

        # Load features
        exp_feat_dict, unip_feat_dict, ess_label_df = load_features(
            args, config, cancer, cell_line, gene_token_dict, gene_info_dict, scgpt_embed
        )

        # Load PPI graph if not ablated
        ppi_edge_index = None
        if not args.no_ppi and exp_feat_dict and unip_feat_dict:
            ppi_edge_index = load_ppi_graph(config, gene_info_table, gene_token_dict, exp_feat_dict)
            print(f"PPI edges: {ppi_edge_index.shape[1] if ppi_edge_index is not None else 0}")

        # Build class dicts
        _, gene_score_dict, train_gene_class_dict, _ = build_gene_class_dicts(
            ess_label_df, args.exclude_pan
        )

        # Resolve paths
        dataset_path = config["datasets"][cancer]["train"]
        dataset_path = dataset_path.replace("{cell_line}", cell_line)

        # Setup output directory
        output_base = config["output"].get(cancer, config["output"][args.cancer])
        train_history = "_".join(args.train_cancers[:i + 1])
        suffix = get_suffix(args, cell_line, train_history)
        output_dir = f"{output_base}/{suffix}"
        os.makedirs(output_dir, exist_ok=True)

        output_prefix = "ess_pred_with_omics"
        training_args = {
            "evaluation_strategy": "no",
            "save_strategy": "steps",
            "save_steps": 42,
            "num_train_epochs": args.epochs,
        }

        # Train
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
            ppi_edge_index=ppi_edge_index,
        )

        cc.train_all_data(
            model_directory=base_model,
            exp_feat_dict=exp_feat_dict,
            unip_feat_dict=unip_feat_dict,
            ppi_edge_index=ppi_edge_index,
            prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
            id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
            output_directory=output_dir,
            output_prefix=output_prefix,
        )

        # Use this model as base for next cancer
        base_model = f"{output_dir}/geneformer_geneClassifier_{output_prefix}"

    return base_model


def test_model(args, config, gene_token_dict, gene_info_table, gene_info_dict, scgpt_embed):
    """Test trained model."""
    torch.cuda.empty_cache()

    cancer = args.cancer
    cell_line = args.cell_line

    # Load features
    exp_feat_dict, unip_feat_dict, ess_label_df = load_features(
        args, config, cancer, cell_line, gene_token_dict, gene_info_dict, scgpt_embed
    )

    # Load PPI graph if not ablated
    ppi_edge_index = None
    if not args.no_ppi and exp_feat_dict and unip_feat_dict:
        ppi_edge_index = load_ppi_graph(config, gene_info_table, gene_token_dict, exp_feat_dict)

    # Build class dicts
    _, gene_score_dict, _, test_gene_class_dict = build_gene_class_dicts(
        ess_label_df, args.exclude_pan
    )

    # Resolve paths
    cl = cell_line or config["labels"][cancer].split("/")[-1].split("_")[0]
    dataset_path_test = config["datasets"][cancer]["test"]
    dataset_path_test = dataset_path_test.replace("{cell_line}", cell_line).replace("{sample}", args.sample)

    # Setup output directory
    output_base = config["output"][cancer]

    if args.multi_cancer:
        train_history = "_".join(args.train_cancers)
        test_suffix = f"test_multi_{args.layers}l_{cl}_sample{args.sample}"
        train_suffix = get_suffix(args, args.train_cell_lines[-1], train_history)
        last_cancer = args.train_cancers[-1]
        finetuned_base = (
            f"{config['output'].get(last_cancer, output_base)}/{train_suffix}"
        )
    else:
        test_suffix = f"test_{get_suffix(args, cl)}"
        train_suffix = get_suffix(args, cl)
        finetuned_base = (
            f"{output_base}/{train_suffix}"
        )

    output_dir = f"{output_base}/{test_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_list = [f"{finetuned_base}/checkpoint-{42 * p}" for p in range(1, 200)]

    # Prepare test data
    output_prefix = "ess_pred_with_omics"
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
    print(f"Eval data size: {len(eval_data)}")

    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    ensembl_gene_dict = gene_info_table.set_index("ensembl_id")["gene_name"].to_dict()

    # Load gene summaries for hit@n evaluation
    gene_summaries = {}
    cancer_summary_map = {"gbm": "u87", "cesc": "hela", "luad": "a549"}
    if cancer in cancer_summary_map:
        cell = cancer_summary_map[cancer]
        for week in ["3w", "4w"]:
            key = f"{cell}_{week}"
            path = config["gene_summaries"].get(key)
            if path:
                gene_summaries[week] = pd.read_csv(path, sep="\t")[["id", "neg|rank", "neg|p-value", "neg|fdr"]]

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
                ppi_edge_index=ppi_edge_index,
            )
        except Exception as e:
            print(f"Skipping {ckp}: {e}")
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
            .apply(lambda g: g.sort_values("y_score_pos", ascending=False)[["y_score_pos", "y_score_neg"]][:100].mean(), include_groups=False)
            .reset_index()
        )
        new_df.to_csv(f"{ckp_dir}/mean_score.csv", index=False)

        torch.cuda.empty_cache()

        # Hit@n evaluation
        if cancer not in ["brca", "coad"] and gene_summaries:
            for week, summary_df in gene_summaries.items():
                rank_score = summary_df.merge(new_df, left_on="id", right_on="gene_n_list", how="left")
                rank_score = rank_score[["id", "neg|rank", "y_score_pos"]]

                cor_dict = {}
                for r in range(2, 100):
                    cor_dict[r] = cal_corscores(
                        rank_score.dropna().sort_values("y_score_pos", ascending=False)[:r]
                    )

                cor_df = pd.DataFrame({
                    "rank": range(2, 100),
                    f"hitn_topn_{week}": [cor_dict[r]["hitn_topn"] for r in range(2, 100)],
                })
                area = [
                    np.trapz(cor_df[f"hitn_topn_{week}"][:rr], cor_df["rank"][:rr])
                    / np.trapz(cor_df["rank"][:rr], cor_df["rank"][:rr])
                    if rr > 0 else 0
                    for rr in range(98)
                ]
                cor_df[f"area_{week}"] = area
                cor_df.to_csv(f"{ckp_dir}/hitn_topn_{week}.csv")


def main():
    args = parse_args()
    config = load_config(args.config)

    # Load shared data
    gene_token_dict = load_token_dict()
    gene_info_table, gene_info_dict = load_gene_info(config)
    scgpt_embed = load_scgpt_embeddings(config) if not args.no_protein else {}

    # Print configuration
    print(f"\n=== CancerFormer Configuration ===")
    print(f"Mode: {'Multi-cancer' if args.multi_cancer else 'Single-cancer'}")
    print(f"Ablations: exp={not args.no_exp}, protein={not args.no_protein}, ppi={not args.no_ppi}")
    if args.multi_cancer:
        print(f"Training sequence: {' -> '.join(args.train_cancers)}")
    else:
        print(f"Cancer: {args.cancer}, Cell line: {args.cell_line or 'default'}")

    # ===== TRAINING =====
    if not args.only_test:
        if args.multi_cancer:
            train_multi_cancer(args, config, gene_token_dict, gene_info_table, gene_info_dict, scgpt_embed)
        else:
            base_model = config["models"][f"geneformer_{args.layers}l"]
            cell_line = args.cell_line or config["labels"][args.cancer].split("/")[-1].split("_")[0]
            train_single_cancer(
                args, config, args.cancer, cell_line, base_model,
                gene_token_dict, gene_info_table, gene_info_dict, scgpt_embed
            )

    # ===== TESTING =====
    if not args.only_train:
        test_model(args, config, gene_token_dict, gene_info_table, gene_info_dict, scgpt_embed)


if __name__ == "__main__":
    main()
