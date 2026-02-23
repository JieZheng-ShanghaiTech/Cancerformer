"""
EssClassifier: subclass of geneformer.Classifier for gene essentiality prediction.

This replaces the need to modify geneformer/classifier.py directly.
Key changes from the standard Classifier:
- Accepts gene_score_dict for regression labels
- Passes feature dicts (exp, unip, gat, ppi) through prepare_data and training
- Uses custom model type 'GeneEssClasifier' with BertForEssGeneClassification
- Adds ParameterClippingCallback for GAT stability
- Uses cosine LR scheduler
- Skips validate_and_clean_cols (handled by patched version)
"""

import logging
import os
import pickle
import subprocess
from pathlib import Path

from geneformer import Classifier
from geneformer import classifier_utils as cu
from geneformer import perturber_utils as pu
from geneformer import DataCollatorForGeneClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments

from essgene.models import ParameterClippingCallback
from essgene.geneformer_patches.perturber_utils_patch import load_model
from essgene.geneformer_patches.classifier_utils_patch import (
    label_classes_with_scores,
    prep_gene_classifier_all_data_with_scores,
    prep_gene_classifier_split_with_scores,
    get_default_train_args_cosine,
)

logger = logging.getLogger(__name__)


class EssClassifier(Classifier):
    """
    Extended Classifier for gene essentiality prediction.

    Adds gene_score_dict support and feature dict passthrough.
    """

    def __init__(
        self,
        classifier="gene",
        gene_class_dict=None,
        gene_score_dict=None,
        max_ncells=None,
        freeze_layers=0,
        training_args=None,
        num_crossval_splits=0,
        forward_batch_size=100,
        nproc=16,
        **kwargs,
    ):
        super().__init__(
            classifier=classifier,
            gene_class_dict=gene_class_dict,
            max_ncells=max_ncells,
            freeze_layers=freeze_layers,
            training_args=training_args,
            num_crossval_splits=num_crossval_splits,
            forward_batch_size=forward_batch_size,
            nproc=nproc,
            **kwargs,
        )
        # Store gene_score_dict mapped to token IDs
        if gene_score_dict is not None:
            self.gene_score_dict = {
                self.gene_token_dict.get(k): v
                for k, v in gene_score_dict.items()
                if self.gene_token_dict.get(k) is not None
            }
        else:
            self.gene_score_dict = {}

    def prepare_data(
        self,
        input_data_file,
        output_directory,
        output_prefix,
        exp_feat_dict=None,
        unip_feat_dict=None,
        gat_feat_dict=None,
        ppi_edge_index=None,
    ):
        """
        Prepare data with score labels and feature dicts.

        Overrides parent to use label_classes_with_scores which adds
        both class labels and regression scores.
        """
        # Load and filter data
        data = pu.load_and_filter(
            self.filter_data, self.nproc, input_data_file
        )
        # Downsample
        data = cu.downsample_and_shuffle(
            data, self.max_ncells, self.max_ncells_per_class, self.cell_state_dict
        )
        # Label with both classes and scores
        data, id_class_dict = label_classes_with_scores(
            self.classifier, data, self.gene_class_dict,
            self.gene_score_dict, self.nproc
        )

        # Save labeled data and id_class_dict
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        labeled_path = output_path / f"{output_prefix}_labeled.dataset"
        data.save_to_disk(str(labeled_path))

        id_class_dict_path = output_path / f"{output_prefix}_id_class_dict.pkl"
        with open(id_class_dict_path, "wb") as f:
            pickle.dump(id_class_dict, f)

        return data, id_class_dict

    def train_all_data(
        self,
        model_directory,
        prepared_input_data_file,
        id_class_dict_file,
        output_directory,
        output_prefix,
        exp_feat_dict=None,
        unip_feat_dict=None,
        gat_feat_dict=None,
        ppi_edge_index=None,
    ):
        """
        Train on all data (no cross-validation) with feature dicts.

        Overrides parent to pass feature dicts to model loading and
        use score-based data preparation.
        """
        # Load id_class_dict
        with open(id_class_dict_file, "rb") as f:
            id_class_dict = pickle.load(f)
        class_id_dict = {v: k for k, v in id_class_dict.items()}
        num_classes = cu.get_num_classes(id_class_dict)

        # Load prepared data (already filtered and labeled)
        train_data = pu.load_and_filter(None, self.nproc, prepared_input_data_file)
        train_data = train_data.shuffle(seed=42)

        # Train
        return self.train_classifier(
            model_directory=model_directory,
            num_classes=num_classes,
            train_data=train_data,
            eval_data=None,
            output_directory=output_directory,
            predict=False,
            exp_feat_dict=exp_feat_dict,
            unip_feat_dict=unip_feat_dict,
            gat_feat_dict=gat_feat_dict,
            ppi_edge_index=ppi_edge_index,
        )

    def train_classifier(
        self,
        model_directory,
        num_classes,
        train_data,
        eval_data,
        output_directory,
        predict=False,
        exp_feat_dict=None,
        unip_feat_dict=None,
        gat_feat_dict=None,
        ppi_edge_index=None,
    ):
        """
        Fine-tune model with custom model type and feature dicts.

        Key differences from parent:
        - Uses 'GeneEssClasifier' model type
        - Passes feature dicts to load_model
        - Adds ParameterClippingCallback
        - Uses cosine LR scheduler
        - Skips validate_and_clean_cols
        """
        if (self.no_eval is True) and (eval_data is not None):
            logger.warning("no_eval set to True; training without evaluation.")
            eval_data = None

        if (self.classifier == "gene") and (predict is True):
            logger.warning("Predictions during training not available for gene classifiers.")
            predict = False

        # Make output directory
        os.makedirs(output_directory, exist_ok=True)

        # Determine model type
        if self.classifier == "cell":
            model_type = "CellClassifier"
        elif self.classifier == "gene":
            model_type = "GeneEssClasifier"
        elif self.classifier == "reg":
            model_type = "GeneEssRegClasifier"
        else:
            model_type = "GeneClassifier"

        # Load model with feature dicts
        model = load_model(
            model_type, num_classes, model_directory, "train",
            exp_feat_dict, unip_feat_dict, gat_feat_dict, ppi_edge_index,
        )

        # Get training args with cosine scheduler
        def_training_args, def_freeze_layers = get_default_train_args_cosine(
            model, self.classifier, train_data, output_directory
        )

        if self.training_args is not None:
            def_training_args.update(self.training_args)

        logging_steps = max(1, round(
            len(train_data) / def_training_args["per_device_train_batch_size"] / 10
        ))
        def_training_args["logging_steps"] = logging_steps
        def_training_args["output_dir"] = output_directory

        training_args_init = TrainingArguments(**def_training_args)

        # Freeze layers
        freeze_layers = self.freeze_layers if self.freeze_layers > 0 else def_freeze_layers
        if freeze_layers > 0:
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        # Setup callbacks
        callbacks = [ParameterClippingCallback()]

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=DataCollatorForGeneClassification(),
            train_dataset=train_data,
            eval_dataset=eval_data,
            callbacks=callbacks,
        )

        # Train
        trainer.train()

        # Save model
        model_output_dir = Path(output_directory) / (
            f"geneformer_{self.classifier}Classifier_{self.training_args.get('output_prefix', 'model')}"
            if self.training_args else f"geneformer_{self.classifier}Classifier"
        )
        trainer.save_model(str(model_output_dir))

        return trainer
