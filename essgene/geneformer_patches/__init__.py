"""
Patches for the standard geneformer package to support gene essentiality prediction.

These patches allow using a standard `pip install geneformer` installation
without modifying the package source code directly.

Usage:
    from essgene.geneformer_patches import EssClassifier
    from essgene.geneformer_patches.classifier_utils_patch import (
        label_classes_with_scores,
        prep_gene_classifier_all_data_with_scores,
    )
"""

from .classifier_patch import EssClassifier

__all__ = ["EssClassifier"]
