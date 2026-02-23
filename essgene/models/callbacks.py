"""Training callbacks and custom loss functions."""

import torch
import torch.nn as nn
from transformers import TrainerCallback


class ParameterClippingCallback(TrainerCallback):
    """Callback to clip GAT attention parameters after each optimizer step."""

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            for name, module in model.named_modules():
                if hasattr(module, 'attn_l') and hasattr(module, 'attn_r'):
                    with torch.no_grad():
                        module.attn_l.clamp_(-0.1, 0.1)
                        module.attn_r.clamp_(-0.1, 0.1)
        return control


class RelativeMSELoss(nn.Module):
    """MSE loss relative to target magnitude."""

    def forward(self, output, target):
        return torch.mean(((output - target) / target) ** 2)
