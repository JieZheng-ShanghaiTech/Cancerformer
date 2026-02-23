"""Extended evaluation metrics for binary gene essentiality classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
)


def py_softmax(x):
    """Compute softmax for a 1D array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_metrics_extended(y_pred, y_true, logits_list, num_classes, labels):
    """
    Compute classification metrics with extended binary metrics.

    Returns AUPRC, max F1, balanced accuracy, Cohen's kappa, and recall
    in addition to standard confusion matrix, macro F1, accuracy, and ROC.
    """
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    roc_metrics = None

    if num_classes == 2:
        y_score = [py_softmax(item)[1] for item in logits_list]

        # AUPRC
        precis, recalls, thresholds = precision_recall_curve(y_true, y_score)
        auprc = auc(recalls, precis)

        # Max F1
        f1_scores = 2 * (precis * recalls) / (precis + recalls)
        max_f1 = np.nanmax(f1_scores)

        # Additional metrics
        bacc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        recall_val = recall_score(y_true, y_pred)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_wt = len(tpr)
        roc_auc = auc(fpr, tpr)
        roc_metrics = {
            "fpr": fpr,
            "tpr": tpr,
            "interp_tpr": interp_tpr,
            "auc": roc_auc,
            "tpr_wt": tpr_wt,
            "auprc": auprc,
            "max_f1": max_f1,
            "balanced_accuracy": bacc,
            "kappa": kappa,
            "recall": recall_val,
        }

    return conf_mat, macro_f1, acc, roc_metrics
