"""Traditional ML baseline methods for gene essentiality classification."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.svm import SVC


def prepare_baseline_features(exp_feat_dict, unip_feat_dict, gene_token_dict, ess_label_df):
    """
    Prepare feature matrix and labels for baseline methods.

    Uses the same features as the main method (expression PCA + scGPT embeddings).
    """
    gene_info_dict_rev = {v: k for k, v in gene_token_dict.items()}

    features = []
    labels = []
    gene_ids = []
    train_test = []

    for _, row in ess_label_df.iterrows():
        ensembl = row["ensembl_id"]
        token_id = gene_token_dict.get(ensembl)
        if token_id is None:
            continue

        exp_feat = exp_feat_dict.get(token_id)
        unip_feat = unip_feat_dict.get(token_id)

        if exp_feat is None or unip_feat is None:
            continue

        feat = np.concatenate([np.array(exp_feat), np.array(unip_feat)])
        features.append(feat)
        labels.append(int(row["Bool"]))
        gene_ids.append(ensembl)
        train_test.append(row.get("train_test", "train"))

    X = np.array(features)
    y = np.array(labels)
    splits = np.array(train_test)

    return X, y, splits, gene_ids


def evaluate_model(y_true, y_pred, y_prob=None):
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        precis, recalls, _ = precision_recall_curve(y_true, y_prob)
        metrics["auprc"] = auc(recalls, precis)
    return metrics


def run_baselines(X, y, splits):
    """
    Run all baseline methods and return results.

    Args:
        X: feature matrix [n_genes, n_features]
        y: binary labels [n_genes]
        splits: train/test split labels [n_genes]

    Returns:
        results: dict of {method_name: metrics_dict}
    """
    train_mask = splits == "train"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
    except ImportError:
        pass

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results[name] = evaluate_model(y_test, y_pred, y_prob)
        print(f"  {name}: {results[name]}")

    return results
