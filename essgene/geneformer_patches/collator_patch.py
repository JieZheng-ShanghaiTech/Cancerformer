"""Patched DataCollator that handles pre-tensorized data correctly."""

import torch
from geneformer import DataCollatorForGeneClassification as _BaseCollator


class DataCollatorForGeneClassificationPatched(_BaseCollator):
    """
    Patched version of DataCollatorForGeneClassification.

    The original uses torch.tensor(v, dtype=torch.int64) which fails when
    v is already a tensor. This version uses v.clone().detach() instead.
    """

    def __call__(self, examples):
        # Call parent to get the batch dict
        batch = super().__call__(examples)
        # Fix tensor conversion: use clone().detach() instead of torch.tensor()
        return {k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.int64)
                for k, v in batch.items()}
