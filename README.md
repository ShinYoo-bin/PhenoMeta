# PhenoMeta: Phenotype-guided Metacell Learning for Single-cell Phenotype Prediction

**PhenoMeta** is a phenotype-aware metacell learning framework that compresses scRNA-seq data to enable computationally efficient patient-level phenotype prediction while preserving predictive performance and biological interpretability.

<p align="center">
  <img src="pipeline.svg" width="100%" alt="PhenoMeta Overview">
</p>

## 1. Preparation

### 1.1 Hardware requirements

GPU acceleration is recommended for scGPT embedding and large datasets. CPU
execution is available with `--device cpu`, but can be substantially slower.

| Dataset | Cells | Recommended GPU | Recommended RAM |
| :------ | ----: | --------------: | --------------: |
| COVID | 26,947 | <4 GB | <4 GB |
| Kidney | 225,177 | <16 GB | <32 GB |
| Cardio | 592,689 | <24 GB | <48 GB |
| Pelka | 370,115 | <16 GB | <32 GB |

These are reference ranges for the datasets and configurations used in this
study; requirements vary with preprocessing, batch size, and model settings.

### 1.2 Environment

```bash
git clone https://github.com/ShinYoo-bin/PhenoMeta.git
cd PhenoMeta
conda env create -f environment.yml
conda activate phenometa
```

Alternatively, install the core dependencies in an existing Python 3.10
environment:

```bash
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CPU or CUDA platform if
needed.

### 1.3 scGPT

PhenoMeta uses [scGPT](https://github.com/bowang-lab/scGPT) as its default
foundation-model backend:

```bash
git clone https://github.com/bowang-lab/scGPT.git external/scGPT
```

Download a compatible pretrained human checkpoint from the official scGPT
model zoo and place `args.json`, `best_model.pt`, and `vocab.json` in
`external/scGPT_checkpoint/`. Override these defaults with
`--scgpt-source-dir`, `--scgpt-checkpoint-dir`, `SCGPT_SOURCE_DIR`, or
`SCGPT_CHECKPOINT_DIR`.

## 2. Data

The source datasets are not redistributed with this repository:

- **COVID:** [Impaired local intrinsic immunity to SARS-CoV-2 infection in severe COVID-19](https://singlecell.broadinstitute.org/single_cell/study/SCP1289/impaired-local-intrinsic-immunity-to-sars-cov-2-infection-in-severe-covid-19)
- **Cardio:** [Single-nuclei profiling of human dilated and hypertrophic cardiomyopathy](https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy)
- **Kidney:** [Human kidney single-cell atlas](https://cellxgene.cziscience.com/collections/0f528c8a-a25c-4840-8fa3-d156fa11086f)
- **Pelka:** [A Single Cell Atlas of MMRd and MMRp Colorectal Cancer (GEO: GSE178341)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178341)

### 2.1 Preprocessing and splitting

[`data/prepare_data.py`](data/prepare_data.py) expects an existing AnnData
`.h5ad` file; it does not download data or convert MTX, CSV, or other source
formats. The input may be stored anywhere. `./data/raw` is used below only as
an example and is intentionally excluded from Git.

```bash
mkdir -p data/raw
# Place or link the converted AnnData file at data/raw/covid.h5ad,
# or replace --input with its actual path.
python data/prepare_data.py \
  --input ./data/raw/covid.h5ad \
  --output-dir ./data/processed/covid \
  --dataset covid \
  --stage all
```

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--input` | required | Existing input `.h5ad` file. |
| `--output-dir` | required | Destination for the processed file and splits. |
| `--dataset` | required | Metadata profile; this study uses `covid`, `cardio`, `kidney`, and `pelka`. |
| `--stage` | `all` | Run `preprocess`, `split`, or both (`all`). |
| `--patient-col`, `--label-col`, `--cell-type-col` | dataset-specific | Override source metadata columns. |
| `--n-folds` | `5` | Number of patient-level folds per repeat. |
| `--seeds` | `0 1 2 3 4` | Seeds defining the repeated splits. |
| `--train-ratio`, `--val-ratio`, `--test-ratio` | `0.6`, `0.2`, `0.2` | Patient-level split ratios. |

The script standardizes dataset-specific metadata to `patient`, `label`, and
`cell_type_annotation`; filters and normalizes expression; selects 3,000 highly
variable genes; and creates patient-disjoint, stratified splits. The default is
five repeats of five 60/20/20 train/validation/test splits.

Use `--stage preprocess` or `--stage split` to run one stage only. Source column
names can be overridden with `--patient-col`, `--label-col`, and
`--cell-type-col`. See `python data/prepare_data.py --help` for all options.

The generated split layout is:

```text
data/processed/<dataset>/splits/
  repeat_0/
    fold_0_train.h5ad
    fold_0_val.h5ad
    fold_0_test.h5ad
    ...
  split_config.json
  split_manifest.csv
```

Each split file must contain a non-negative cell-by-gene matrix in `adata.X`,
patient IDs in `adata.obs["patient"]`, phenotype labels in
`adata.obs["label"]`, unique cell and gene identifiers, and gene symbols in
`var_names` or a supported `adata.var` column. Patients must be disjoint across
train, validation, and test sets.

## 3. Method overview

PhenoMeta performs four steps:

1. scGPT embeddings are generated patient by patient and cached.
2. A patient-aware graph autoencoder learns a compact cell representation.
3. Prototypes are initialized, refined, and merged before phenotype-guided
   cell-to-metacell assignment.
4. Patient-specific metacells are pooled by a size-aware attention predictor.

The submission model uses **ELU** in the assignment-gate value branch,
metacell encoder, attention network, and phenotype classifier. The
multiplicative assignment gate retains its sigmoid activation. After joint
training, routing is frozen and the patient-level predictor is optimized
separately.

## 4. Training and prediction

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --input-dir ./data/processed/<dataset>/splits \
  --output-dir ./results/<dataset>/joint \
  --dataset <dataset> \
  --foundation-model scgpt \
  --scgpt-source-dir ./external/scGPT \
  --scgpt-checkpoint-dir ./external/scGPT_checkpoint \
  --scgpt-batch-size 200 \
  --device cuda \
  --predictor-only-init joint \
  --repeats 0,1,2,3,4 \
  --folds 0,1,2,3,4
```

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--input-dir` | required | Root containing repeat/fold split files. |
| `--output-dir` | required | Training and prediction output directory. |
| `--dataset` | input directory name | Dataset name recorded in outputs. |
| `--patient-col`, `--label-col` | `patient`, `label` | AnnData observation columns. |
| `--foundation-model` | `scgpt` | Embedding backend (`scgpt`, `uce`, or `precomputed`). |
| `--scgpt-source-dir` | `external/scGPT` | Local scGPT source tree. |
| `--scgpt-checkpoint-dir` | `external/scGPT_checkpoint` | Pretrained checkpoint directory. |
| `--scgpt-batch-size` | `2000` | scGPT inference batch size; reported runs used `200`. |
| `--use-graph-ae` | enabled | Enable graph-autoencoder representation learning. |
| `--device` | `cuda` | PyTorch device; use `cpu` for CPU execution. |
| `--epochs` | `100` | Maximum joint-training epochs. |
| `--predictor-only-epochs` | `50` | Maximum frozen-routing predictor epochs. |
| `--predictor-only-init` | `reset` | Predictor initialization; reported runs used `joint`. |
| `--repeats`, `--folds` | auto-discovered | Comma-separated repeat and fold IDs. |
| `--seed` | `3407` | Base random seed. |

For a smoke test, set both `--repeats` and `--folds` to `0`. Completed folds
are skipped automatically; use `--rerun-completed`,
`--overwrite-foundation-cache`, or `--overwrite-graph-ae` when recomputation is
required. Run `python train.py --help` for the complete option list.

## 5. Optional global correction

The reported final pipeline optionally combines frozen PhenoMeta probabilities
with an L2-regularized pseudobulk classifier:

```text
softmax(log(P_PhenoMeta) + alpha * log(P_pseudobulk))
```

Regularization and `alpha` are selected on validation data only. Because
`alpha=0` is included and preferred on ties, the method falls back to the
uncorrected PhenoMeta prediction when correction does not improve validation
AUROC.

```bash
python global_correction.py \
  --input-dir ./data/processed/<dataset>/splits \
  --attention-output-dir ./results/<dataset>/joint \
  --output-dir ./results/<dataset>/final \
  --cache-dir ./results/<dataset>/cache/pseudobulk \
  --patient-col patient \
  --repeats 0,1,2,3,4 \
  --folds 0,1,2,3,4
```

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--input-dir` | required | Root containing original split files. |
| `--attention-output-dir` | required | Completed `train.py` output directory. |
| `--output-dir` | required | Corrected prediction output directory. |
| `--cache-dir` | `<output-dir>/cache/pseudobulk` | Persistent patient-mean cache. |
| `--patient-col` | `patient` | Patient identifier column. |
| `--repeats`, `--folds` | auto-discovered | Comma-separated repeat and fold IDs. |
| `--seed` | `3407` | Base random seed. |
| `--alphas` | `0 0.05 0.1 0.25 0.5` | Candidate probability-mixing weights. |

This step is not required to train PhenoMeta. Use `joint/` for uncorrected
outputs and `final/` for the corrected pipeline. See
`python global_correction.py --help` for all options.

## 6. Outputs

The main output directories are:

```text
results/<dataset>/joint/
  assignments/       cell-to-metacell assignments
  metacells/         patient-specific metacell AnnData files
  predictions/       train, validation, and test probabilities
  checkpoints/       model checkpoints
  results_folds.csv  fold-level metrics
  results_repeats.csv
  config_used.json
  resource.csv
  summary.txt

results/<dataset>/final/
  predictions/       corrected test predictions
  selection.csv      selected ridge C and alpha per fold
  results_folds.csv
  config_used.json
  summary.txt
```

## 7. Repository structure

```text
PhenoMeta/
  phenometa/            model, training, embedding, and data utilities
  data/prepare_data.py  preprocessing and split generation
  tests/                unit tests
  train.py              main training entry point
  global_correction.py  optional validation-gated correction
  requirements.txt
  environment.yml
```

## 8. Reproducibility

- The reported configuration used five repeats, five folds, seed `3407`, scGPT
  batch size `200`, and `--predictor-only-init joint`.
- Validation data are used for early stopping and correction selection only;
  test labels are not used for model or hyperparameter selection.
- Preserve `split_manifest.csv` and `config_used.json` with reported results.
