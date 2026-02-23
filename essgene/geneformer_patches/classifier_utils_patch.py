"""
Patched classifier utility functions for gene essentiality prediction.

These replace modified functions from geneformer.classifier_utils to support:
- gene_score_dict for regression labels
- Feature dict parameters (exp, unip, gat)
- Cosine LR scheduler
"""

import logging

import numpy as np
from geneformer import perturber_utils as pu
from geneformer.classifier_utils import (
    downsample_and_shuffle,
    label_gene_classes,
)

logger = logging.getLogger(__name__)


def label_gene_scores(example, class_id_dict, gene_score_dict):
    """Map each token in example to its essentiality score."""
    return [
        gene_score_dict.get(token_id, -100)
        for token_id in example["input_ids"]
    ]


def label_classes_with_scores(classifier, data, gene_class_dict, gene_score_dict, nproc):
    """
    Label data with scores in the 'labels' field.

    Unlike the original label_classes(), this puts scores directly into
    the 'labels' field (not class IDs), matching the original lifespan implementation.
    """
    if classifier == "gene":
        def if_contains_label(example):
            a = pu.flatten_list(gene_class_dict.values())
            b = example["input_ids"]
            return not set(a).isdisjoint(b)

        data = data.filter(if_contains_label, num_proc=nproc)
        label_set = gene_class_dict.keys()

        if len(data) == 0:
            logger.error("No cells remain after filtering for target genes.")
            raise RuntimeError("No cells after filtering")
    else:
        label_set = set(data["label"])

    class_id_dict = dict(zip(label_set, range(len(label_set))))
    id_class_dict = {v: k for k, v in class_id_dict.items()}

    def classes_to_ids(example):
        if classifier == "cell":
            example["label"] = class_id_dict[example["label"]]
        elif classifier == "gene":
            # Put scores directly into labels field (not class IDs)
            example["labels"] = label_gene_scores(example, class_id_dict, gene_score_dict)
        return example

    data = data.map(classes_to_ids, num_proc=nproc)
    return data, id_class_dict


def prep_gene_classifier_split_with_scores(
    data, targets, labels, scores, train_index, eval_index,
    max_ncells, iteration_num, num_proc
):
    """
    Prepare train/eval splits for gene classifier with score-based labels.

    Uses scores (not class labels) for relabeling, enabling regression training.
    """
    targets = np.array(targets)
    scores = np.array(scores)
    targets_train, targets_eval = targets[train_index], targets[eval_index]
    labels_train, labels_eval = scores[train_index], scores[eval_index]
    label_dict_train = dict(zip(targets_train, labels_train))
    label_dict_eval = dict(zip(targets_eval, labels_eval))

    def if_contains_train_label(example):
        return not set(targets_train).isdisjoint(example["input_ids"])

    def if_contains_eval_label(example):
        return not set(targets_eval).isdisjoint(example["input_ids"])

    logger.info(f"Filtering training data for genes in split {iteration_num}")
    train_data = data.filter(if_contains_train_label, num_proc=num_proc)
    logger.info(f"Filtered {round((1-len(train_data)/len(data))*100)}%; {len(train_data)} remain")

    logger.info(f"Filtering evaluation data for genes in split {iteration_num}")
    eval_data = data.filter(if_contains_eval_label, num_proc=num_proc)
    logger.info(f"Filtered {round((1-len(eval_data)/len(data))*100)}%; {len(eval_data)} remain")

    train_data = downsample_and_shuffle(train_data, max_ncells, None, None)
    eval_data = downsample_and_shuffle(eval_data, max_ncells, None, None)

    def train_classes_to_ids(example):
        example["scores"] = [
            label_dict_train.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    def eval_classes_to_ids(example):
        example["scores"] = [
            label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    train_data = train_data.map(train_classes_to_ids, num_proc=num_proc)
    eval_data = eval_data.map(eval_classes_to_ids, num_proc=num_proc)
    return train_data, eval_data


def prep_gene_classifier_all_data_with_scores(
    data, targets, labels, scores, max_ncells, num_proc,
    exp_feat_dict=None, unip_feat_dict=None, gat_feat_dict=None
):
    """
    Prepare all data for gene classifier training with score-based labels.

    Uses scores for label_dict instead of class labels.
    """
    targets = np.array(targets)
    scores = np.array(scores)
    label_dict_train = dict(zip(targets, scores))

    def if_contains_train_label(example):
        return not set(targets).isdisjoint(example["input_ids"])

    logger.info("Filtering training data for genes to classify.")
    train_data = data.filter(if_contains_train_label, num_proc=num_proc)
    logger.info(f"Filtered {round((1-len(train_data)/len(data))*100)}%; {len(train_data)} remain")

    train_data = downsample_and_shuffle(train_data, max_ncells, None, None)

    def train_classes_to_ids(example):
        example["labels"] = [
            label_dict_train.get(token_id, -100) for token_id in example["input_ids"]
        ]
        return example

    train_data = train_data.map(train_classes_to_ids, num_proc=num_proc)
    return train_data


def validate_and_clean_cols_with_scores(train_data, eval_data, classifier):
    """
    Validate and clean dataset columns, keeping 'scores' column.

    Unlike the original, this keeps the 'scores' column needed for
    regression training.
    """
    if classifier == "cell":
        label_col = "label"
    elif classifier == "gene":
        label_col = "labels"

    cols_to_keep = [label_col, "input_ids", "length", "scores"]

    if label_col not in train_data.column_names:
        logger.error(f"train_data must contain column {label_col} with class labels.")
        raise RuntimeError(f"Missing column: {label_col}")

    other_cols = [c for c in train_data.features.keys() if c not in cols_to_keep]
    train_data = train_data.remove_columns(other_cols)

    if eval_data is not None:
        if label_col not in eval_data.column_names:
            logger.error(f"eval_data must contain column {label_col} with class labels.")
            raise RuntimeError(f"Missing column: {label_col}")
        other_cols = [c for c in eval_data.features.keys() if c not in cols_to_keep]
        eval_data = eval_data.remove_columns(other_cols)

    return train_data, eval_data


def get_default_train_args_cosine(model, classifier, data, output_dir):
    """
    Get default training args with cosine LR scheduler.

    Same as the original get_default_train_args but uses cosine scheduler.
    """
    from geneformer import perturber_utils as pu

    num_layers = pu.quant_layers(model)
    freeze_layers = 0
    batch_size = 12

    if classifier == "cell":
        epochs = 10
        evaluation_strategy = "epoch"
        load_best_model_at_end = True
    else:
        epochs = 1
        evaluation_strategy = "no"
        load_best_model_at_end = False

    if num_layers == 6:
        default_training_args = {
            "learning_rate": 5e-5,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 500,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }
    elif num_layers == 12:
        default_training_args = {
            "learning_rate": 3e-5,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 500,
            "per_device_train_batch_size": batch_size // 2,
            "per_device_eval_batch_size": batch_size // 2,
        }
    else:
        default_training_args = {
            "learning_rate": 5e-5,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 500,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
        }

    training_args = {
        "num_train_epochs": epochs,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": evaluation_strategy,
        "logging_steps": max(1, round(len(data) / batch_size / 8)),
        "save_strategy": "epoch",
        "group_by_length": False,
        "length_column_name": "length",
        "disable_tqdm": False,
        "weight_decay": 0.001,
        "load_best_model_at_end": load_best_model_at_end,
    }
    training_args.update(default_training_args)
    return training_args, freeze_layers
