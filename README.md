# PhenoMeta: Phenotype-guided Metacell Learning for Single-cell Phenotype Prediction

**PhenoMeta** is an annotation-free framework for patient-level phenotype
prediction from single-cell RNA sequencing data. It learns patient-specific
metacells through phenotype-guided cell-to-prototype assignment and aggregates
the resulting metacells with a size-aware attention predictor.

The pipeline consists of four main stages:

1. **Foundation embedding** – Encode individual cells with a pretrained
   single-cell foundation model (scGPT by default).
2. **Graph representation learning** – Compress the foundation embeddings with
   a patient-aware graph autoencoder.
3. **Phenotype-guided metacell learning** – Initialize, refine, and merge
   prototypes before jointly learning cell assignments and phenotype prediction.
4. **Prediction and correction** – Train a frozen-routing size-aware attention
   predictor and optionally apply validation-gated pseudobulk correction.

```text
scRNA-seq split files
  -> patient-wise scGPT embeddings
  -> graph autoencoder embeddings
  -> prototype initialization and refinement
  -> phenotype-guided metacell assignment
  -> size-aware attention prediction
  -> frozen-routing predictor optimization
  -> optional validation-gated global correction
```

## 1. Preparation

### 1.1 Hardware requirements

GPU acceleration is strongly recommended for scGPT embedding and model
training. CPU execution is supported through `--device cpu`, but is generally
impractical for large single-cell datasets.

The following values are observed process-level peaks from the experiments used
in this study. They are provided as reference values rather than strict minimum
requirements.

| Dataset | Peak process RAM | Peak reserved GPU memory |
| :------ | ---------------: | -----------------------: |
| Cardio | 8.5 GB | 6.4 GB |
| COVID | 6.4 GB | 2.1 GB |
| Kidney | 14.4 GB | 5.1 GB |
| Pelka | 6.9 GB | 6.3 GB |

Actual requirements depend on cell count, gene count, scGPT batch size, and
the selected model parameters. Additional memory headroom is recommended.

### 1.2 Environment setup

Clone this repository and create the Conda environment:

```bash
git clone <PHENOMETA_REPOSITORY_URL>
cd PhenoMeta
conda env create -f environment.yml
conda activate phenometa
```

Alternatively, install the core Python dependencies in an existing Python 3.10
environment:

```bash
python -m pip install -r requirements.txt
```

The provided dependency versions correspond to the environment used to verify
this implementation. Install the PyTorch build appropriate for the local CPU or
CUDA platform when the default package is not compatible with the system.

### 1.3 Download scGPT and a pretrained checkpoint

PhenoMeta uses the official [scGPT](https://github.com/bowang-lab/scGPT)
implementation as its default foundation-model backend. Clone scGPT separately
and follow its installation instructions:

```bash
git clone https://github.com/bowang-lab/scGPT.git external/scGPT
```

Download a compatible pretrained human checkpoint from the official scGPT model
zoo. The checkpoint directory must contain:

```text
scGPT_checkpoint/
  args.json
  best_model.pt
  vocab.json
```

By default, PhenoMeta looks for these resources in `external/scGPT` and
`external/scGPT_checkpoint`. Override them with the corresponding CLI arguments
or `SCGPT_SOURCE_DIR` and `SCGPT_CHECKPOINT_DIR` environment variables. The
pretrained weights are not distributed with this repository and remain subject
to the original scGPT license and terms.

## 2. Input data

The source datasets are not redistributed with this repository. The integrated
[`data/prepare_data.py`](data/prepare_data.py) script standardizes raw AnnData
files and generates patient-disjoint evaluation splits. It contains default
metadata mappings for BPD, Cardio, COVID, Kidney, Lupus, Parkinson, and Pelka;
source column names can be overridden with `--patient-col`, `--label-col`, and
`--cell-type-col`.

### 2.1 Preprocessing and split generation

Run preprocessing and splitting together:

```bash
python data/prepare_data.py \
  --input ./data/raw/covid.h5ad \
  --output-dir ./data/processed/covid \
  --dataset covid \
  --stage all
```

The preprocessing stage filters genes, normalizes total counts, applies a
log1p transformation, selects 3,000 highly variable genes, and standardizes
metadata to `patient`, `label`, and `cell_type_annotation`. The split stage is
patient-disjoint and stratified by phenotype label. By default, it creates five
repeats of five 60/20/20 train/validation/test splits using seeds `0 1 2 3 4`.

Use `--stage preprocess` to create only the processed `.h5ad` file or
`--stage split` to split an already processed file. Run
`python data/prepare_data.py --help` for all available options.

The command above produces:

```text
data/processed/covid/
  covid_preprocessed.h5ad
  covid_preprocessed.json
  splits/
    repeat_0/
      fold_0_train.h5ad
      fold_0_val.h5ad
      fold_0_test.h5ad
      ...
    split_config.json
    split_manifest.csv
```

For Parkinson, the study configuration samples 30% of each patient's cells
once before splitting and removes `adata.raw` from the split files. Other
configured datasets retain all cells by default. Override these settings with
`--sample-fraction` and `--no-drop-raw` when needed.

`split_manifest.csv` records the exact patient assignment, phenotype label,
repeat, fold, random seed, cell-sampling count, and output file. Retain this
manifest with reported results to make the evaluation splits reproducible.

### 2.2 Directory layout

PhenoMeta expects pre-split AnnData files arranged by repeat and fold:

```text
INPUT_DIR/
  repeat_0/
    fold_0_train.h5ad
    fold_0_val.h5ad          # optional explicit validation split
    fold_0_test.h5ad
    fold_1_train.h5ad
    fold_1_val.h5ad
    fold_1_test.h5ad
    ...
  repeat_1/
    ...
```

If `fold_<N>_val.h5ad` is absent, validation patients are sampled from the
corresponding training split. The default validation fraction is `0.25`.

### 2.3 AnnData requirements

Each `.h5ad` file must satisfy the following requirements:

- `adata.X` contains a non-negative cell-by-gene expression matrix.
- `adata.obs["patient"]` contains patient identifiers by default.
- `adata.obs["label"]` contains patient phenotype labels by default.
- Every patient has exactly one phenotype label within a split.
- Cell identifiers (`obs_names`) and gene identifiers (`var_names`) are unique.
- Patients are disjoint across the train, validation, and test sets.
- Gene symbols are available in `var_names` or a supported `adata.var` column.

Use `--patient-col`, `--label-col`, or `--scgpt-gene-col` if the input uses
different column names.

## 3. PhenoMeta pipeline

### 3.1 Foundation embedding

Cells are encoded patient by patient with scGPT. Genes are matched against the
checkpoint vocabulary, and the resulting cell embeddings are cached for reuse.
The cache includes metadata checks to prevent incompatible inputs from being
silently reused.

### 3.2 Graph autoencoder

A k-nearest-neighbor graph is constructed from the foundation embeddings. The
graph autoencoder optimizes reconstruction, graph smoothness, and contrastive
objectives to produce the latent representation used for prototype assignment.

### 3.3 Phenotype-guided metacell learning

PhenoMeta initializes prototypes with approximate farthest-point sampling and
iteratively refines soft cell-to-prototype assignments. Low-usage prototypes
are removed, and highly similar prototypes with compatible usage are merged.

The final routing stage uses top-k candidate prototypes, a learned assignment
gate, and straight-through Gumbel-Softmax during training. Raw expression is
aggregated according to the learned assignments to construct patient-specific
metacells.

### 3.4 Patient-level prediction

Metacells are encoded and pooled with size-aware attention. The attention weight
combines metacell abundance with a learned relevance score. The submission model
uses **ELU** in the assignment-gate value branch, metacell encoder, attention
network, and phenotype classifier. The multiplicative gate retains its sigmoid
activation.

After joint training, routing is frozen and the patient-level predictor is
optimized separately. The `joint` initialization mode continues from the
jointly trained predictor and was used in the reported experiments.

## 4. Training and prediction

### 4.1 Generic usage

Run the complete foundation embedding, graph autoencoder, metacell learning, and
predictor pipeline with `train.py`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --input-dir ./data/<dataset>/splits \
  --output-dir ./results/<dataset>/joint \
  --dataset <dataset> \
  --patient-col patient \
  --label-col label \
  --foundation-model scgpt \
  --scgpt-source-dir ./external/scGPT \
  --scgpt-checkpoint-dir ./external/scGPT_checkpoint \
  --scgpt-batch-size 200 \
  --device cuda \
  --predictor-only-init joint \
  --repeats 0,1,2,3,4 \
  --folds 0,1,2,3,4
```

For a one-fold smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --input-dir ./data/<dataset>/splits \
  --output-dir ./results/<dataset>/joint \
  --dataset <dataset> \
  --scgpt-source-dir ./external/scGPT \
  --scgpt-checkpoint-dir ./external/scGPT_checkpoint \
  --scgpt-batch-size 200 \
  --device cuda \
  --predictor-only-init joint \
  --repeats 0 \
  --folds 0
```

Completed folds are skipped automatically. Use `--rerun-completed` to rerun
them, `--overwrite-foundation-cache` to recompute foundation embeddings, and
`--overwrite-graph-ae` to retrain graph-autoencoder caches.

### 4.2 Main arguments

| Argument | Type | Default | Required | Description |
| :------- | :--- | :------ | :------: | :---------- |
| `--input-dir` | path | – | Yes | Root containing the repeat/fold `.h5ad` files. |
| `--output-dir` | path | – | Yes | Output directory for training artifacts and predictions. |
| `--dataset` | string | input directory name | No | Dataset name recorded in result files. |
| `--patient-col` | string | `patient` | No | Patient identifier column in `adata.obs`. |
| `--label-col` | string | `label` | No | Phenotype label column in `adata.obs`. |
| `--foundation-model` | string | `scgpt` | No | Foundation embedding backend. |
| `--scgpt-source-dir` | path | `external/scGPT` | For scGPT | Path to the cloned scGPT source tree. |
| `--scgpt-checkpoint-dir` | path | `external/scGPT_checkpoint` | For scGPT | Directory containing `args.json`, `best_model.pt`, and `vocab.json`. |
| `--scgpt-batch-size` | integer | `2000` | No | Batch size for scGPT inference; `200` was used in the reported experiments. |
| `--use-graph-ae` | boolean | `true` | No | Enable graph-autoencoder representation learning. |
| `--ae-latent-dim` | integer | `32` | No | Graph-autoencoder latent dimension. |
| `--epochs` | integer | `100` | No | Maximum joint-training epochs. |
| `--predictor-only-epochs` | integer | `50` | No | Maximum frozen-routing predictor epochs. |
| `--predictor-only-init` | choice | `reset` | No | Initialize the frozen predictor from `reset` or `joint` weights. |
| `--pooling` | choice | `size_aware_attention` | No | Patient-level metacell pooling strategy. |
| `--repeats` | CSV integers | auto | No | Repeat identifiers, such as `0,1,2,3,4`. |
| `--folds` | CSV integers | auto | No | Fold identifiers, such as `0,1,2,3,4`. |
| `--device` | string | `cuda` | No | PyTorch device. |
| `--seed` | integer | `3407` | No | Base random seed. |

Run `python train.py --help` for the complete list of model and optimization
arguments.

## 5. Validation-gated global correction

The reported final pipeline includes an optional pseudobulk correction step.
`global_correction.py` computes patient-level mean expression, trains an
L2-regularized logistic-regression model, and combines its probabilities with
the frozen PhenoMeta attention probabilities:

```text
softmax(log(P_PhenoMeta) + alpha * log(P_pseudobulk))
```

The ridge strength and mixing weight are selected using validation data only.
The candidate set always includes `alpha=0`, and ties in validation AUROC favor
the smaller mixing weight. Therefore, the correction falls back to the
PhenoMeta prediction when it does not improve validation AUROC.

Run the correction after `train.py` completes:

```bash
python global_correction.py \
  --input-dir ./data/<dataset>/splits \
  --attention-output-dir ./results/<dataset>/joint \
  --output-dir ./results/<dataset>/final \
  --cache-dir ./results/<dataset>/cache/pseudobulk \
  --patient-col patient \
  --repeats 0,1,2,3,4 \
  --folds 0,1,2,3,4
```

| Argument | Type | Default | Required | Description |
| :------- | :--- | :------ | :------: | :---------- |
| `--input-dir` | path | – | Yes | Root containing the original split files. |
| `--attention-output-dir` | path | – | Yes | Completed `train.py` output directory. |
| `--output-dir` | path | – | Yes | Directory for corrected predictions and summaries. |
| `--cache-dir` | path | `<output>/cache/pseudobulk` | No | Persistent patient-mean cache. |
| `--patient-col` | string | `patient` | No | Patient identifier column. |
| `--repeats` | CSV integers | all discovered | No | Repeat identifiers to process. |
| `--folds` | CSV integers | all discovered | No | Fold identifiers to process. |
| `--seed` | integer | `3407` | No | Base random seed. |
| `--cs` | float list | predefined grid | No | Logistic-regression regularization candidates. |
| `--alphas` | float list | `0 0.05 0.1 0.25 0.5` | No | Probability-mixing candidates. |

This correction is not required to train the PhenoMeta neural network. Use the
`joint` outputs for the uncorrected PhenoMeta result and the `final` outputs for
the complete corrected pipeline.

## 6. Outputs

### 6.1 Training outputs

```text
results/<dataset>/joint/
  assignments/                 cell-to-metacell assignments and gate scores
  checkpoints/                 joint-training checkpoints
  metacells/                   exported patient-specific metacell AnnData files
  predictions/                 train, validation, and test probabilities
  predictor_only_checkpoints/  frozen-routing predictor checkpoints
  predictor_only_history/      predictor optimization histories
  refinement_logs/             prototype refinement diagnostics
  training_history/            joint-training histories
  config_used.json             complete command-line configuration
  resource.csv                 sampled CPU, memory, I/O, and GPU usage
  results_folds.csv            fold-level metrics
  results_repeats.csv          repeat-level summaries
  summary.json
  summary.txt
```

### 6.2 Global-correction outputs

```text
results/<dataset>/final/
  predictions/       corrected test predictions
  config_used.json   correction configuration and processed jobs
  selection.csv      selected ridge C and mixing alpha for each fold
  results_folds.csv  corrected, ridge-only, and attention-only metrics
  summary.json
  summary.txt
```

The `final/summary.txt` file reports three result groups:

- **Combined** – PhenoMeta plus the validation-selected correction.
- **Pseudobulk ridge only** – the global-expression classifier alone.
- **Frozen attention baseline** – uncorrected PhenoMeta predictions.

## 7. Repository structure

```text
PhenoMeta/
  phenometa/
    config.py             model configuration
    data.py               split discovery and patient-bag loading
    graph_autoencoder.py  graph representation learning
    model.py              prototype assignment and attention predictor
    patient_cache.py      reusable patient-level cache management
    scgpt_embedding.py    scGPT embedding generation
    training.py           joint and frozen-routing optimization
  data/
    prepare_data.py       preprocessing and patient-level split generation
  tests/                  unit tests
  train.py                main training and prediction entry point
  global_correction.py    validation-gated pseudobulk correction
  requirements.txt        core Python dependencies
  environment.yml         Conda environment definition
```

## 8. Reproducibility notes

- The submission model uses fixed ELU activations in the PhenoMeta assignment
  and prediction networks.
- Graph-autoencoder hidden layers retain ReLU activations, and the assignment
  gate retains its sigmoid branch.
- The reported benchmark configuration used five repeats, five folds, seed
  `3407`, scGPT batch size `200`, and `--predictor-only-init joint`.
- Validation data are used for early stopping and correction selection only;
  test labels are not used for model or hyperparameter selection.
- Input splits should be prepared before running this repository and must be
  patient-disjoint.