"""
Patched model loading function supporting custom essentiality model types.

Replaces the modified load_model() from geneformer.perturber_utils.
"""

from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
)

from essgene.models import BertForEssGeneClassification, BertForEssRegGeneClassification


def load_model(model_type, num_classes, model_directory, mode,
               exp_feat_dict=None, unip_feat_dict=None,
               gat_feat_dict=None, ppi_edge_index=None):
    """
    Load a model by type, supporting custom essentiality classifiers.

    Extends the standard geneformer load_model with:
    - 'GeneEssClasifier': BertForEssGeneClassification (with PPI-GAT)
    - 'GeneEssRegClasifier': BertForEssRegGeneClassification (regression)
    """
    output_hidden_states = (mode == "eval")

    if model_type == "Pretrained":
        model = BertForMaskedLM.from_pretrained(
            model_directory,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
    elif model_type == "GeneClassifier":
        model = BertForTokenClassification.from_pretrained(
            model_directory,
            num_labels=num_classes,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
    elif model_type == "CellClassifier":
        model = BertForSequenceClassification.from_pretrained(
            model_directory,
            num_labels=num_classes,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
        )
    elif model_type == "GeneEssClasifier":
        model = BertForEssGeneClassification.from_pretrained(
            model_directory,
            num_labels=num_classes,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            exp_feat_dict=exp_feat_dict,
            unip_feat_dict=unip_feat_dict,
            ppi_edge_index=ppi_edge_index,
        )
    elif model_type == "GeneEssRegClasifier":
        model = BertForEssRegGeneClassification.from_pretrained(
            model_directory,
            num_labels=num_classes,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            exp_feat_dict=exp_feat_dict,
            unip_feat_dict=unip_feat_dict,
            gat_feat_dict=gat_feat_dict,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model
