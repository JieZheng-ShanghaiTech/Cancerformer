"""BERT-based gene essentiality regression classifier."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


def _trans_dict_featmat(dic, d, max_ind=25426):
    """Convert a {token_id: feature_vector} dict to a dense feature matrix."""
    feat_mat = torch.zeros([max_ind, d])
    for k in dic.keys():
        feat_mat[k] = torch.tensor(dic[k])
    return feat_mat


class BertForEssRegGeneClassification(BertForTokenClassification):
    """
    BERT model for gene essentiality regression + classification.

    Uses expression and protein features concatenated with BERT hidden states,
    with a combined CE + MSE loss.
    """

    def __init__(self, config, exp_feat_dict=None, unip_feat_dict=None, gat_feat_dict=None):
        super().__init__(config)

        self.exp_dim = 0
        self.unip_dim = 0
        self.gat_dim = 0

        if exp_feat_dict:
            self.exp_dim = 64
            self.exp_feat_mat = _trans_dict_featmat(exp_feat_dict, 64)
        if unip_feat_dict:
            first_value = next(iter(unip_feat_dict.values()))
            self.unip_dim = len(first_value)
            self.unip_feat_mat = _trans_dict_featmat(unip_feat_dict, self.unip_dim)
        if gat_feat_dict:
            first_value = next(iter(gat_feat_dict.values()))
            self.gat_dim = len(first_value)
            self.gat_feat_mat = _trans_dict_featmat(gat_feat_dict, self.gat_dim)

        total_dim = config.hidden_size + self.exp_dim + self.unip_dim + self.gat_dim
        self.ess_classifier1 = nn.Linear(total_dim, config.hidden_size)
        self.ess_classifier3 = nn.Linear(config.hidden_size, config.num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

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
        batch_n = sequence_output.shape[0]

        concated_feat = sequence_output
        if self.exp_dim != 0:
            exp_features = torch.zeros([sequence_output.shape[0], sequence_output.shape[1], self.exp_dim])
            for b in range(batch_n):
                exp_features[b, :, :] = self.exp_feat_mat[input_ids[b].cpu()]
            concated_feat = torch.cat((concated_feat, exp_features.to(sequence_output.device)), 2)
        if self.unip_dim != 0:
            unip_features = torch.zeros([sequence_output.shape[0], sequence_output.shape[1], self.unip_dim])
            for b in range(batch_n):
                unip_features[b, :, :] = self.unip_feat_mat[input_ids[b].cpu()]
            concated_feat = torch.cat((concated_feat, unip_features.to(sequence_output.device)), 2)
        if self.gat_dim != 0:
            gat_features = torch.zeros([sequence_output.shape[0], sequence_output.shape[1], self.gat_dim])
            for b in range(batch_n):
                gat_features[b, :, :] = self.gat_feat_mat[input_ids[b].cpu()]
            concated_feat = torch.cat((concated_feat, gat_features.to(sequence_output.device)), 2)

        lin_out = self.relu(self.ess_classifier1(concated_feat))
        logits = self.ess_classifier3(lin_out)
        logits = self.softmax(logits)

        bool_label = torch.tensor(labels.view(-1) > 0, dtype=labels.dtype)
        bool_label[labels.view(-1) == -100] = -100
        bool_label = bool_label.long()
        mask = bool_label != -100
        pred_scores = logits.view(-1, self.num_labels)[:, 1][mask]
        mask_label = labels.view(-1)[mask]

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), bool_label) + MSELoss()(pred_scores, mask_label)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
