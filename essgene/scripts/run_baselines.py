"""
Run traditional ML baselines for gene essentiality classification.

Usage:
    python -m essgene.scripts.run_baselines --cancer cesc --cell_line caski
"""

import argparse
import json

from essgene.data import (
    load_config,
    load_expression_features,
    load_gene_info,
    load_essentiality_labels,
    prepare_feature_dicts,
)
from essgene.data.loader import load_scgpt_embeddings, load_token_dict
from essgene.baselines.ml_baselines import prepare_baseline_features, run_baselines


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML baselines for gene essentiality")
    parser.add_argument("--cancer", type=str, default="cesc")
    parser.add_argument("--cell_line", type=str, default="caski")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    gene_token_dict = load_token_dict()
    gene_info_table, gene_info_dict = load_gene_info(config)
    scgpt_embed = load_scgpt_embeddings(config)
    exp_feat_dict_raw = load_expression_features(args.cancer, config)
    ess_label_df = load_essentiality_labels(args.cancer, config, args.cell_line)

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

    X, y, splits, gene_ids = prepare_baseline_features(
        exp_feat_dict, unip_feat_dict, gene_token_dict, ess_label_df
    )
    print(f"Features shape: {X.shape}, Labels: {sum(y==1)} essential, {sum(y==0)} non-essential")

    results = run_baselines(X, y, splits)

    print("\n=== Baseline Results ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    output_path = f"{config['output'][args.cancer]}/baseline_results_{args.cancer}_{args.cell_line}.json"
    with open(output_path, "w") as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
