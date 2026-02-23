# CancerFormer: A CRISPR Screen-benchmarked Multimodal AI Platform for Prediction of Cancer Dependencies in Patient-derived Organoids

[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **CancerFormer**, a multimodal deep learning framework for predicting cancer gene dependencies. CancerFormer integrates single-cell transcriptomics, bulk RNA-seq, protein embeddings, and protein-protein interaction networks to identify essential genes for cancer cell survival, validated against CRISPR screening data from patient-derived organoids.

![Graph abstract](graph_abstract.png)

## Table of Contents

- [Abstract](#abstract)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [License](#license)

## Abstract

Dissection of cancer dependencies is the central topic of cancer research. Recent advances in artificial intelligence (AI) have provided opportunities for rapid prediction of cancer essential genes. However, these AI models are often limited by the incapability of leveraging multimodal information or insufficient benchmarks, leading to low success rates in physiologically relevant practice.

Here, we developed **CancerFormer**, a multimodal deep learning framework that integrates single-cell RNA sequencing (scRNA-seq), TCGA transcriptomic profiles, and protein-protein interaction (PPI) networks to predict cancer gene essentiality. By employing a Transformer architecture to capture gene functional context and Graph Neural Networks to embed topological structures of PPI networks, CancerFormer overcomes the generalization limitations of existing methods.

Using experimental results of CRISPR screens from multiple cancer cell lines (HeLa, A549, and U-87MG), we demonstrated that CancerFormer consistently outperformed state-of-the-art baseline models under both gene-wise and sample-wise cross-validation splits as measured by multiple evaluation metrics. In subsequent applications, CancerFormer demonstrated strong generalization ability, achieving a **90% experimental verification rate** for top candidates in colorectal cancer HCT116 cells.

Most importantly, without pre-training on patient-derived organoids (PDOs) data, CancerFormer successfully captured inter-patient heterogeneity in PDOs and revealed a context-specific metabolic dependency on oxidative phosphorylation pathways in 3D culture compared to 2D cell lines. Functional assays on top predicted targets confirmed their essentiality in PDO growth.

This study established CancerFormer as a rigorously benchmarked multimodal AI model for predicting cancer dependencies with physiological relevance.


## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Create conda environment
conda create -n essgene python=3.10
conda activate essgene

# Install dependencies
cd essgene
pip install -r requirements.txt
```

## Quick Start

### Training CancerFormer (Full Model)

Train on a single cancer type with all features:

```bash
python -m essgene.scripts.train_cancerformer \
    --cancer gbm \
    --layers 6 \
    --only_train
```

### Ablation Studies

Test contribution of different components:

```bash
# Without PPI-GAT
python -m essgene.scripts.train_cancerformer --cancer gbm --no_ppi --only_train

# Without expression features
python -m essgene.scripts.train_cancerformer --cancer gbm --no_exp --only_train

# Without protein features
python -m essgene.scripts.train_cancerformer --cancer gbm --no_protein --only_train

# Geneformer only (all ablations)
python -m essgene.scripts.train_cancerformer --cancer gbm --no_exp --no_protein --no_ppi --only_train
```

### Multi-Cancer Training

Sequential training across multiple cancer types:

```bash
python -m essgene.scripts.train_cancerformer \
    --multi_cancer \
    --train_cancers cesc gbm luad \
    --train_cell_lines siha gbm luad \
    --cancer gbm \
    --cell_line gbm \
    --only_train
```

### Testing

Evaluate a trained model:

```bash
python -m essgene.scripts.train_cancerformer \
    --cancer gbm \
    --cell_line gbm \
    --only_test
```

## Key Arguments

### Basic Arguments
- `--cancer`: Cancer type (gbm, cesc, luad, brca, coad)
- `--cell_line`: Cell line name (e.g., caski, hela, siha)
- `--freeze_layers`: Number of BERT layers to freeze (default: 0)
- `--epochs`: Training epochs (default: 1)
- `--subsample_size`: Max cells per gene (default: 10,000)
- `--only_train` / `--only_test`: Run only training or testing phase

### Ablation Arguments
- `--no_exp`: Disable expression features
- `--no_protein`: Disable protein/scGPT embeddings
- `--no_ppi`: Disable PPI-GAT module

### Multi-Cancer Arguments
- `--multi_cancer`: Enable multi-cancer sequential training
- `--train_cancers`: List of cancer types to train on
- `--train_cell_lines`: Corresponding cell lines for each cancer

## Data Requirements

### Required Data Files

The project expects the following data structure (configured in `config/paths.yaml`):

```
/path/to/resourses
├── TCGA
│   ├── TCGA-{CANCER}_mRNA.csv          # Bulk RNA-seq expression
│   └── processed/complete_ess_ppi.csv  # PPI graph
├── single_cell_dataset/
│   ├── {cancer}/
│   │   ├── {cell_line}_label.csv       # Essentiality labels
│   │   └── train/*.dataset             # Tokenized datasets
│   └── scgpt_gene_embeddings.npy       # scGPT protein embeddings
└── CRISPR/                        # Validation data
    ├── A549/                           # LUAD validation
    ├── Hela/                           # CESC validation
    └── u87-mg/                         # GBM validation
```

### Data Preparation

1. **Tokenized Datasets**: Pre-tokenized single-cell data using Geneformer tokenizer
2. **Expression Features**: Log-transformed bulk RNA-seq, PCA-reduced to 64 dimensions
3. **Protein Embeddings**: scGPT embeddings mapped to gene token IDs
4. **PPI Graph**: Edge list format (source, target) with gene token IDs
5. **Labels**: CSV with gene names and essentiality scores (0.0-1.0)

## Output Structure

Training outputs are saved to:
```
cancerformer/single_model/{cancer}/train_{layers}l_{cell_line}_freez_{freeze_layers}_{suffix}/
├── checkpoint-{step}/              # Model checkpoints (every 42 steps)
├── ess_pred_with_omics_labeled.dataset/  # Prepared training data
├── pred_dict.pkl                   # Predictions with gene names
└── mean_score.csv                  # Aggregated scores (top 100 cells/gene)
```

Suffix indicates ablations:
- `_full`: All features enabled
- `_noppi`: Without PPI-GAT
- `_noexp`: Without expression features
- `_noprotein`: Without protein embeddings

## License

This project is licensed under the MIT License - see the LICENSE file for details.