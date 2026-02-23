"""BERT-based gene essentiality classifier with PPI-GAT integration."""

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from .gat_layers import GATLayer


def _trans_dict_featmat(dic, d, max_ind=None):
    """Convert a {token_id: feature_vector} dict to a dense feature matrix."""
    if max_ind is None:
        max_ind = max(dic.keys()) if dic else 25426
    max_ind = max(max_ind, 25426)
    feat_mat = torch.zeros([max_ind + 1, d])
    for k in dic.keys():
        if 0 <= k <= max_ind:
            feat_mat[k] = torch.tensor(dic[k])
    return feat_mat


class BertForEssGeneClassification(BertForTokenClassification):
    """
    BERT model for gene essentiality classification.

    Extends BertForTokenClassification with:
    - Expression feature integration (PCA of bulk RNA-seq)
    - Protein/gene embedding integration (scGPT embeddings)
    - Optional PPI graph attention network (GAT) module
    - Combined cross-entropy + MSE loss for joint classification and regression
    """

    def __init__(self, config, exp_feat_dict=None, unip_feat_dict=None, ppi_edge_index=None):
        super().__init__(config)

        self.exp_dim = 0
        self.unip_dim = 0
        self.ppi_gat_dim = 0

        # Determine maximum token ID from PPI edge_index if available
        max_token_id = 25426
        if ppi_edge_index is not None:
            max_token_id = max(max_token_id, ppi_edge_index.max().item())

        # Expression features
        if exp_feat_dict:
            self.exp_dim = len(next(iter(exp_feat_dict.values())))
            max_id = max(max(exp_feat_dict.keys()), max_token_id)
            self.exp_feat_mat = _trans_dict_featmat(exp_feat_dict, self.exp_dim, max_id)

        # Protein/gene embedding features
        if unip_feat_dict:
            self.unip_dim = len(next(iter(unip_feat_dict.values())))
            max_id = max(max(unip_feat_dict.keys()), max_token_id)
            self.unip_feat_mat = _trans_dict_featmat(unip_feat_dict, self.unip_dim, max_id)

        # GAT on PPI graph
        self.ppi_edge_index = ppi_edge_index
        use_simple_gnn = os.getenv('USE_SIMPLE_GNN', '0') == '1'
        use_stable_gat = os.getenv('USE_STABLE_GAT', '0') == '1'

        if self.exp_dim > 0 and self.unip_dim > 0 and ppi_edge_index is not None:
            self.ppi_gat_in_dim = self.exp_dim + self.unip_dim + config.hidden_size
            self.ppi_gat_dim = 512
            self.gat = GATLayer(
                self.ppi_gat_in_dim, self.ppi_gat_dim,
                num_heads=1, dropout=0.1,
                use_simple_gnn=use_simple_gnn,
                use_stable_gat=use_stable_gat,
            )

            # Register gradient clipping hooks for original GAT
            if not use_simple_gnn and not use_stable_gat:
                if hasattr(self.gat, 'attn_l') and hasattr(self.gat, 'attn_r'):
                    def clip_grad(grad):
                        return torch.clamp(grad, min=-0.01, max=0.01) if grad is not None else grad
                    self.gat.attn_l.register_hook(clip_grad)
                    self.gat.attn_r.register_hook(clip_grad)

        # Classifier head
        self.ess_classifier1 = nn.Linear(config.hidden_size + self.ppi_gat_dim, config.hidden_size)
        self.ess_classifier3 = nn.Linear(config.hidden_size, config.num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def get_node_features(self, input_ids):
        """Get per-token expression + protein features."""
        batch_n = input_ids.shape[0]
        exp_feat = torch.zeros([input_ids.shape[0], input_ids.shape[1], self.exp_dim])
        unip_feat = torch.zeros([input_ids.shape[0], input_ids.shape[1], self.unip_dim])
        max_exp_idx = self.exp_feat_mat.shape[0] - 1
        max_unip_idx = self.unip_feat_mat.shape[0] - 1
        for b in range(batch_n):
            ids_b = input_ids[b].cpu()
            exp_feat[b, :, :] = self.exp_feat_mat[torch.clamp(ids_b, 0, max_exp_idx)]
            unip_feat[b, :, :] = self.unip_feat_mat[torch.clamp(ids_b, 0, max_unip_idx)]
        return torch.cat((exp_feat.to(input_ids.device), unip_feat.to(input_ids.device)), 2)

    def batch_ppi_gat(self, all_node_feats, input_ids):
        """Apply GAT over PPI subgraph for each batch item."""
        out_batch = []
        for i in range(all_node_feats.shape[0]):
            node_feats = all_node_feats[i]
            node_gat_feats = torch.zeros(node_feats.size(0), self.ppi_gat_dim, device=node_feats.device)

            if self.ppi_edge_index is not None and node_feats.size(0) > 1:
                try:
                    edge_index = self.ppi_edge_index.to(node_feats.device)
                    input_ids_list = input_ids[i].cpu().tolist()
                    idx_map = {gid: n for n, gid in enumerate(input_ids_list)}
                    seq_len = node_feats.size(0)

                    edges_in_seq = []
                    edge_src = edge_index[0].cpu().tolist()
                    edge_tgt = edge_index[1].cpu().tolist()
                    for ss, tt in zip(edge_src, edge_tgt):
                        if ss in idx_map and tt in idx_map:
                            src_idx, tgt_idx = idx_map[ss], idx_map[tt]
                            if 0 <= src_idx < seq_len and 0 <= tgt_idx < seq_len:
                                edges_in_seq.append((src_idx, tgt_idx))

                    if edges_in_seq:
                        edge_index_batch = torch.tensor(edges_in_seq, dtype=torch.long, device=node_feats.device).t()
                        if edge_index_batch.max().item() < seq_len:
                            node_feats_clean = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
                            node_gat_feats = self.gat(node_feats_clean, edge_index_batch)
                            node_gat_feats = torch.nan_to_num(node_gat_feats, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass  # Fallback to zeros

            out_batch.append(node_gat_feats)
        return torch.stack(out_batch, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs[0])
        batch_n, seq_len = sequence_output.shape[0], sequence_output.shape[1]
        sequence_output = torch.nan_to_num(sequence_output, nan=0.0, posinf=0.0, neginf=0.0)

        node_feats = self.get_node_features(input_ids) if self.exp_dim > 0 and self.unip_dim > 0 else None

        if node_feats is not None and self.ppi_gat_dim > 0:
            node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
            gat_input = torch.cat([sequence_output, node_feats.to(sequence_output.device)], dim=2)
            gat_out = self.batch_ppi_gat(gat_input, input_ids)
            gat_out = torch.nan_to_num(gat_out, nan=0.0, posinf=0.0, neginf=0.0)
            concated_feat = torch.cat([sequence_output, gat_out.to(sequence_output.device)], dim=2)
        else:
            if self.ppi_gat_dim > 0:
                gat_out = torch.zeros((batch_n, seq_len, self.ppi_gat_dim), device=sequence_output.device)
                concated_feat = torch.cat([sequence_output, gat_out], dim=2)
            else:
                concated_feat = sequence_output

        concated_feat = torch.nan_to_num(concated_feat, nan=0.0, posinf=0.0, neginf=0.0)
        lin_out = self.relu(self.ess_classifier1(concated_feat))
        lin_out = torch.nan_to_num(lin_out, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.ess_classifier3(lin_out)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        logits = self.softmax(logits)

        # Loss: CE on binary labels + MSE on regression scores
        # labels contains the scores, bool_label is derived from labels > 0
        bool_label = (labels.view(-1) > 0).to(dtype=labels.dtype, device=logits.device)
        bool_label[labels.view(-1) == -100] = -100
        bool_label = bool_label.long()
        mask = bool_label != -100
        pred_scores = logits.view(-1, self.num_labels)[:, 1][mask]
        mask_label = labels.view(-1)[mask].float()  # Convert to float for MSE loss

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_reg = nn.MSELoss()
            loss = loss_fct(logits.view(-1, self.num_labels), bool_label) + loss_reg(pred_scores, mask_label)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
