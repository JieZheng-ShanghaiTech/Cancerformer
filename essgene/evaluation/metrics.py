"""Evaluation metrics for gene essentiality prediction."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def softmax(x):
    """Compute column-wise softmax."""
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


def cal_corscores(score_df):
    """
    Calculate correlation scores between predicted and actual gene rankings.

    Args:
        score_df: DataFrame with columns [id, neg|rank, pred_score]

    Returns:
        dict with spearman_corr and hitn_topn
    """
    if len(score_df.columns) > 3:
        score_df.columns = ["id", "neg|rank"] + [f"pred_score_{i}" for i in range(len(score_df.columns) - 2)]
    else:
        score_df.columns = ["id", "neg|rank", "pred_score"]
    score_df = score_df.dropna()

    hitn_topn = sum(score_df["neg|rank"].values < len(score_df))
    vector1 = score_df["neg|rank"].values
    vector2 = -score_df["pred_score"].values

    spearman_corr = spearmanr(vector1, vector2)[0]
    tau, _ = kendalltau(vector1, vector2)

    return {"spearman_corr": spearman_corr, "hitn_topn": hitn_topn}


def compute_hitn_topn(new_df, gene_summary_df, output_dir, ckp):
    """
    Compute hit@n metrics across different top-n cutoffs.

    Args:
        new_df: DataFrame with gene_n_list, y_score_pos columns
        gene_summary_df: DataFrame with id, neg|rank columns
        output_dir: directory to save results
        ckp: checkpoint name

    Returns:
        cor_scores_df: DataFrame with hit@n and area metrics
    """
    rank_score = gene_summary_df.merge(
        new_df, left_on="id", right_on="gene_n_list", how="left"
    )[["id", "neg|rank", "y_score_pos"]]

    cor_scores_dict = {}
    for r in range(2, 100):
        cor_scores_dict[r] = cal_corscores(
            rank_score.dropna().sort_values(by="y_score_pos", ascending=False)[:r]
        )

    cor_scores_df = pd.DataFrame()
    cor_scores_df["rank"] = range(2, 100)
    cor_scores_df["hitn_topn"] = [cor_scores_dict[r]["hitn_topn"] for r in range(2, 100)]

    area = [
        np.trapz(cor_scores_df["hitn_topn"][:rr], cor_scores_df["rank"][:rr])
        / np.trapz(cor_scores_df["rank"][:rr], cor_scores_df["rank"][:rr])
        if rr > 0 else 0
        for rr in range(98)
    ]
    cor_scores_df["area"] = area

    if output_dir and ckp:
        cor_scores_df.to_csv(f"{output_dir}/{ckp}/hitn_topn.csv")

    return cor_scores_df
