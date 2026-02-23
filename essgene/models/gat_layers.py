"""GAT layer implementations for PPI graph integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGNNLayer(nn.Module):
    """Simple Graph Neural Network layer as a stable alternative to GAT."""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        if edge_index.size(1) == 0:
            return torch.zeros(num_nodes, self.linear.out_features, device=x.device, dtype=x.dtype)

        src, tgt = edge_index
        valid_mask = (src < num_nodes) & (tgt < num_nodes) & (src >= 0) & (tgt >= 0)
        if valid_mask.sum() == 0:
            return torch.zeros(num_nodes, self.linear.out_features, device=x.device, dtype=x.dtype)

        src = src[valid_mask]
        tgt = tgt[valid_mask]

        x_transformed = self.linear(x)
        out = torch.zeros(num_nodes, x_transformed.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, tgt.unsqueeze(-1).expand(-1, x_transformed.size(1)), x_transformed[src])

        degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        degree.scatter_add_(0, tgt, torch.ones_like(tgt, dtype=x.dtype))
        degree = torch.clamp(degree + 1.0, min=1.0)

        out = out / degree.unsqueeze(-1)
        out = self.dropout(out)
        return out


class StableGATLayer(nn.Module):
    """
    Stable GAT implementation with layer normalization, temperature scaling,
    and conservative parameter initialization.
    """

    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.out_dim = out_dim

        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.attn_weight = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_dim))
        self.temperature = nn.Parameter(torch.ones(1) * (out_dim ** 0.5))
        self.output_norm = nn.LayerNorm(out_dim * num_heads if concat else out_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.attn_weight, a=-0.001, b=0.001)
        with torch.no_grad():
            self.temperature.fill_(self.out_dim ** 0.5)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        x = self.layer_norm(x)

        if edge_index.size(1) == 0:
            out_dim = self.num_heads * self.out_dim if self.concat else self.out_dim
            return torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)

        src, tgt = edge_index
        valid_mask = (src >= 0) & (src < num_nodes) & (tgt >= 0) & (tgt < num_nodes)
        if valid_mask.sum() == 0:
            out_dim = self.num_heads * self.out_dim if self.concat else self.out_dim
            return torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)

        src = src[valid_mask]
        tgt = tgt[valid_mask]

        x_transformed = self.linear(x).view(num_nodes, self.num_heads, self.out_dim)
        attn_weight_clamped = torch.clamp(self.attn_weight, min=-0.01, max=0.01)

        x_src = x_transformed[src]
        x_tgt = x_transformed[tgt]
        x_concat = torch.cat([x_src, x_tgt], dim=-1)

        alpha = (x_concat * attn_weight_clamped).sum(dim=-1)
        alpha = self.leaky_relu(alpha)

        temperature = torch.clamp(self.temperature, min=1.0, max=10.0)
        alpha = torch.clamp(alpha / temperature, min=-10.0, max=10.0)

        alpha_max = alpha.max(dim=0, keepdim=True)[0].detach()
        exp_alpha = torch.exp(alpha - alpha_max)

        attn_sum = torch.zeros(num_nodes, self.num_heads, device=x.device, dtype=exp_alpha.dtype)
        tgt_expanded = tgt.unsqueeze(-1).expand(-1, self.num_heads)
        attn_sum.scatter_add_(0, tgt_expanded, exp_alpha)

        attn_sum = torch.clamp(attn_sum, min=1e-8)
        attn_coef = exp_alpha / attn_sum[tgt]
        attn_coef = self.dropout(attn_coef)

        weighted_feat = attn_coef.unsqueeze(-1) * x_src
        out = torch.zeros(num_nodes, self.num_heads, self.out_dim, device=x.device, dtype=x.dtype)
        tgt_expanded_3d = tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.out_dim)
        out.scatter_add_(0, tgt_expanded_3d, weighted_feat)

        if self.concat:
            out = out.view(num_nodes, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = self.output_norm(out)
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network layer with multiple backend options.

    Args:
        use_simple_gnn: Use SimpleGNNLayer (most stable, no attention)
        use_stable_gat: Use StableGATLayer (attention with stability features)
    """

    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.1, concat=True,
                 use_simple_gnn=False, use_stable_gat=False):
        super().__init__()
        self.use_simple_gnn = use_simple_gnn
        self.use_stable_gat = use_stable_gat

        if use_simple_gnn:
            self.gnn = SimpleGNNLayer(in_dim, out_dim * num_heads if concat else out_dim, dropout)
            return

        if use_stable_gat:
            self.stable_gat = StableGATLayer(in_dim, out_dim, num_heads, dropout, concat)
            return

        self.num_heads = num_heads
        self.concat = concat
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_simple_gnn:
            return
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.uniform_(self.attn_l, a=-0.01, b=0.01)
        nn.init.uniform_(self.attn_r, a=-0.01, b=0.01)

    def forward(self, x, edge_index):
        if self.use_simple_gnn:
            return self.gnn(x, edge_index)
        if self.use_stable_gat:
            return self.stable_gat(x, edge_index)

        num_nodes = x.size(0)

        # Handle NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if edge_index.size(1) == 0:
            out_dim = self.num_heads * self.out_dim if self.concat else self.out_dim
            return torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)

        src, tgt = edge_index
        valid_mask = (src >= 0) & (src < num_nodes) & (tgt >= 0) & (tgt < num_nodes)
        if valid_mask.sum() == 0:
            out_dim = self.num_heads * self.out_dim if self.concat else self.out_dim
            return torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)

        src = src[valid_mask]
        tgt = tgt[valid_mask]

        x = self.linear(x).view(num_nodes, self.num_heads, self.out_dim)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp attention parameters (non-inplace)
        clamp_max = 0.1
        attn_l_clamped = torch.clamp(self.attn_l, min=-clamp_max, max=clamp_max)
        attn_r_clamped = torch.clamp(self.attn_r, min=-clamp_max, max=clamp_max)

        alpha_l = (x * attn_l_clamped).sum(dim=-1)
        alpha_r = (x * attn_r_clamped).sum(dim=-1)
        alpha_l = torch.nan_to_num(alpha_l, nan=0.0, posinf=0.0, neginf=0.0)
        alpha_r = torch.nan_to_num(alpha_r, nan=0.0, posinf=0.0, neginf=0.0)

        alpha = self.leaky_relu(alpha_l[src] + alpha_r[tgt])
        alpha = torch.clamp(alpha, min=-20.0, max=20.0)

        # Stable softmax
        alpha_max = alpha.max(dim=0, keepdim=True)[0]
        exp_alpha = torch.exp(alpha - alpha_max)

        attn_sum = torch.zeros(num_nodes, self.num_heads, device=x.device, dtype=exp_alpha.dtype)
        tgt_expanded = tgt.unsqueeze(-1).expand(-1, self.num_heads)
        attn_sum.scatter_add_(0, tgt_expanded, exp_alpha)

        attn_sum = torch.nan_to_num(attn_sum, nan=1e-9, posinf=1e10, neginf=1e-9)
        attn_sum_per_edge = torch.clamp(attn_sum[tgt], min=1e-10)
        attn_coef = exp_alpha / attn_sum_per_edge

        if torch.isnan(attn_coef).any() or torch.isinf(attn_coef).any():
            attn_coef = torch.ones_like(attn_coef) / num_nodes

        attn_coef = self.dropout(attn_coef)
        x_j = x[src]

        weighted_feat = attn_coef.unsqueeze(-1) * x_j
        out = torch.zeros(num_nodes, self.num_heads, self.out_dim, device=x.device, dtype=x.dtype)
        tgt_expanded_3d = tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.out_dim)
        out.scatter_add_(0, tgt_expanded_3d, weighted_feat)

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if self.concat:
            out = out.view(num_nodes, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out
