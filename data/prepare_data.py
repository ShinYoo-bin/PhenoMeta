#!/usr/bin/env python3
"""Preprocess single-cell data and create patient-level repeated splits.

This script consolidates the preprocessing and split-generation procedures used
for PhenoMeta. It can run either stage independently or execute both stages in
sequence. No project-specific absolute paths are assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass(frozen=True)
class DatasetSpec:
    """Source metadata columns and dataset-specific sampling defaults."""

    patient_col: str
    cell_type_col: str
    label_col: str
    sample_fraction: float = 1.0
    drop_raw: bool = False


DATASET_SPECS = {
    "bpd": DatasetSpec("donor_id", "cell_type", "disease"),
    "cardio": DatasetSpec(
        "patient", "cell_type_annotation", "disease__ontology_label"
    ),
    "covid": DatasetSpec(
        "patient", "original_cell_type_annotation", "disease__ontology_label"
    ),
    "kidney": DatasetSpec("patient", "cell_type_annotation", "past_label"),
    "lupus": DatasetSpec("ind_cov", "ct_cov", "disease_cov"),
    "parkinson": DatasetSpec(
        "donor_id", "cell_type", "disease", sample_fraction=0.3, drop_raw=True
    ),
    "pelka": DatasetSpec(
        "donor_id", "cell_type_coarse_study", "microsatellite_status"
    ),
}


def _temporary_path(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    return Path(temporary)


def atomic_write_h5ad(
    adata: ad.AnnData,
    path: Path,
    *,
    overwrite: bool,
    compression: str,
) -> bool:
    """Write an AnnData file atomically and return whether it was written."""

    path = Path(path)
    if path.exists() and not overwrite:
        print(f"[skip] output already exists: {path}")
        return False
    temporary = _temporary_path(path, ".tmp.h5ad")
    try:
        kwargs = {} if compression == "none" else {"compression": compression}
        adata.write_h5ad(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = _temporary_path(path, ".tmp.csv")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(value: dict, path: Path) -> None:
    path = Path(path)
    temporary = _temporary_path(path, ".tmp.json")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_column(
    adata: ad.AnnData,
    override: str | None,
    standardized: str,
    configured: str,
) -> str:
    if override:
        candidate = override
    elif standardized in adata.obs:
        candidate = standardized
    else:
        candidate = configured
    if candidate not in adata.obs:
        raise KeyError(
            f"required metadata column {candidate!r} is absent; "
            f"available columns: {list(adata.obs.columns)}"
        )
    return candidate


def standardize_metadata(
    adata: ad.AnnData,
    spec: DatasetSpec,
    *,
    patient_col: str | None,
    label_col: str | None,
    cell_type_col: str | None,
) -> tuple[ad.AnnData, dict[str, str]]:
    """Create the standardized patient, label, and cell-type columns."""

    resolved_patient = _source_column(
        adata, patient_col, "patient", spec.patient_col
    )
    resolved_label = _source_column(adata, label_col, "label", spec.label_col)
    resolved_cell_type = _source_column(
        adata, cell_type_col, "cell_type_annotation", spec.cell_type_col
    )

    valid = adata.obs[resolved_patient].notna() & adata.obs[resolved_label].notna()
    removed = int((~valid).sum())
    if removed:
        print(f"[metadata] removing {removed} cells with missing patient or label")
        adata = adata[valid].copy()

    adata.obs["patient"] = adata.obs[resolved_patient].astype(str)
    adata.obs["label"] = adata.obs[resolved_label].astype(str)
    adata.obs["name_label"] = adata.obs[resolved_label].astype(str)
    adata.obs["cell_type_annotation"] = adata.obs[resolved_cell_type].astype(str)
    return adata, {
        "patient": resolved_patient,
        "label": resolved_label,
        "cell_type_annotation": resolved_cell_type,
    }


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> Path:
    """Normalize expression, select highly variable genes, and standardize metadata."""

    if output_path.exists() and not args.overwrite:
        print(f"[preprocess] reusing existing file: {output_path}")
        return output_path

    print(f"[preprocess] reading {input_path}")
    adata = ad.read_h5ad(input_path)
    adata.var_names_make_unique()
    initial_shape = tuple(map(int, adata.shape))
    adata, resolved_columns = standardize_metadata(
        adata,
        spec,
        patient_col=args.patient_col,
        label_col=args.label_col,
        cell_type_col=args.cell_type_col,
    )

    print(f"[preprocess] filtering genes expressed in fewer than {args.min_cells} cells")
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    print(f"[preprocess] normalizing each cell to {args.target_sum:g} total counts")
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    print("[preprocess] applying log1p transformation")
    sc.pp.log1p(adata)

    n_top_genes = min(args.n_top_genes, adata.n_vars)
    if n_top_genes <= 0:
        raise ValueError("--n-top-genes must be positive")
    print(f"[preprocess] selecting {n_top_genes} highly variable genes")
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n_top_genes,
    )
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    # Preserve the behavior of the original preprocessing pipeline.
    adata.uns.pop("log1p", None)

    atomic_write_h5ad(
        adata,
        output_path,
        overwrite=True,
        compression=args.compression,
    )
    config = {
        "dataset": args.dataset,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "initial_shape": initial_shape,
        "final_shape": tuple(map(int, adata.shape)),
        "resolved_columns": resolved_columns,
        "min_cells": args.min_cells,
        "target_sum": args.target_sum,
        "n_top_genes": n_top_genes,
    }
    atomic_write_json(config, output_path.with_suffix(".json"))
    print(f"[preprocess] wrote {output_path} with shape={adata.shape}")
    return output_path


def patient_seed(seed: int, patient: str) -> np.random.SeedSequence:
    digest = hashlib.sha256(str(patient).encode("utf-8")).digest()
    patient_value = int.from_bytes(digest[:8], "little")
    return np.random.SeedSequence([int(seed), patient_value & 0xFFFFFFFF])


def sample_patient_cells(
    adata: ad.AnnData,
    patient_col: str,
    fraction: float,
    seed: int,
) -> tuple[ad.AnnData, list[dict]]:
    """Sample every patient once so all repeats reuse the same selected cells."""

    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")

    selected: list[np.ndarray] = []
    manifest: list[dict] = []
    groups = adata.obs.groupby(patient_col, sort=True, observed=False).indices
    for patient, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        n_sampled = len(indices)
        if not np.isclose(fraction, 1.0):
            n_sampled = max(1, int(round(len(indices) * fraction)))
            rng = np.random.default_rng(patient_seed(seed, str(patient)))
            indices = rng.choice(indices, size=n_sampled, replace=False)
        selected.append(indices)
        manifest.append(
            {
                "patient": str(patient),
                "n_original": len(raw_indices),
                "n_sampled": n_sampled,
                "sample_fraction_realized": n_sampled / len(raw_indices),
            }
        )

    selected_indices = np.sort(np.concatenate(selected))
    if len(selected_indices) == adata.n_obs:
        return adata, manifest
    return adata[selected_indices].copy(), manifest


def resolve_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    """Normalize ratio parts, while requiring fractional inputs to sum to one."""

    values = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("train/validation/test ratios must be positive finite numbers")
    total = float(values.sum())
    if total > 1.0 + 1e-8:
        values /= total
    elif not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            "fractional train/validation/test ratios must sum to 1.0; "
            f"received {train_ratio}, {val_ratio}, {test_ratio}"
        )
    return tuple(map(float, values))



def _patient_table(adata: ad.AnnData) -> pd.DataFrame:
    table = adata.obs[["patient", "label"]].drop_duplicates().reset_index(drop=True)
    duplicated = table["patient"].duplicated(keep=False)
    if duplicated.any():
        patients = table.loc[duplicated, "patient"].unique().tolist()
        raise ValueError(f"patients with multiple labels: {patients[:10]}")
    return table


def _outer_splitter(test_ratio: float, n_folds: int, seed: int):
    if np.isclose(test_ratio, 1.0 / n_folds):
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return StratifiedShuffleSplit(
        n_splits=n_folds,
        test_size=test_ratio,
        random_state=seed,
    )


def create_patient_splits(
    input_path: Path,
    output_dir: Path,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> Path:
    """Create patient-disjoint train/validation/test files and an exact manifest."""

    train_ratio, val_ratio, test_ratio = resolve_ratios(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    print(f"[split] reading {input_path}")
    adata = ad.read_h5ad(input_path)
    adata, resolved_columns = standardize_metadata(
        adata,
        spec,
        patient_col=args.patient_col,
        label_col=args.label_col,
        cell_type_col=args.cell_type_col,
    )

    sample_fraction = (
        spec.sample_fraction if args.sample_fraction is None else args.sample_fraction
    )
    drop_raw = spec.drop_raw if args.drop_raw is None else args.drop_raw
    adata, sampling_rows = sample_patient_cells(
        adata,
        "patient",
        sample_fraction,
        args.sample_seed,
    )
    if drop_raw:
        adata.raw = None
    output_dir.mkdir(parents=True, exist_ok=True)
    patients = _patient_table(adata)
    sample_by_patient = {row["patient"]: row for row in sampling_rows}
    manifest_rows: list[dict] = []

    print(
        f"[split] patients={len(patients)} cells={adata.n_obs} "
        f"ratios={train_ratio:.3f}/{val_ratio:.3f}/{test_ratio:.3f} "
        f"sample_fraction={sample_fraction:g} drop_raw={drop_raw}"
    )

    for repeat, seed in enumerate(args.seeds):
        splitter = _outer_splitter(test_ratio, args.n_folds, seed)
        repeat_dir = output_dir / f"repeat_{repeat}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_val_idx, test_idx) in enumerate(
            splitter.split(patients["patient"], patients["label"])
        ):
            train_val = patients.iloc[train_val_idx].reset_index(drop=True)
            inner_validation_fraction = val_ratio / (train_ratio + val_ratio)
            inner_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=inner_validation_fraction,
                random_state=seed * 1000 + fold,
            )
            train_local, val_local = next(
                inner_splitter.split(train_val["patient"], train_val["label"])
            )
            split_indices = {
                "train": train_val_idx[train_local],
                "val": train_val_idx[val_local],
                "test": test_idx,
            }

            split_summary = []
            for split_name, patient_indices in split_indices.items():
                selected_patients = patients.iloc[patient_indices]["patient"].to_numpy()
                subset = adata[adata.obs["patient"].isin(selected_patients)].copy()
                subset.obs["patient"] = subset.obs["patient"].astype(str)
                subset.obs["label"] = subset.obs["label"].astype(str)
                path = repeat_dir / f"fold_{fold}_{split_name}.h5ad"
                atomic_write_h5ad(
                    subset,
                    path,
                    overwrite=args.overwrite,
                    compression=args.compression,
                )

                relative_path = str(path.relative_to(output_dir))
                for patient in selected_patients:
                    patient = str(patient)
                    row = sample_by_patient[patient]
                    label = patients.loc[
                        patients["patient"] == patient, "label"
                    ].iloc[0]
                    manifest_rows.append(
                        {
                            "dataset": args.dataset,
                            "repeat": repeat,
                            "seed": seed,
                            "fold": fold,
                            "split": split_name,
                            "patient": patient,
                            "label": str(label),
                            "file": relative_path,
                            **row,
                            "sample_seed": args.sample_seed,
                            "train_ratio": train_ratio,
                            "val_ratio": val_ratio,
                            "test_ratio": test_ratio,
                        }
                    )
                split_summary.append(f"{split_name}={len(selected_patients)}")
            print(f"[repeat={repeat} fold={fold}] " + " ".join(split_summary))

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["repeat", "fold", "split", "patient"]
    )
    atomic_write_frame(manifest, output_dir / "split_manifest.csv")
    config = {
        "dataset": args.dataset,
        "input": str(input_path.resolve()),
        "output": str(output_dir.resolve()),
        "resolved_columns": resolved_columns,
        "n_folds": args.n_folds,
        "seeds": args.seeds,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "sample_fraction": sample_fraction,
        "sample_seed": args.sample_seed,
        "drop_raw": drop_raw,
        "compression": args.compression,
        "dataset_defaults": asdict(spec),
    }
    atomic_write_json(config, output_dir / "split_config.json")
    print(f"[split] wrote {output_dir / 'split_manifest.csv'}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .h5ad file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dataset output directory containing the preprocessed file and splits",
    )
    parser.add_argument(
        "--dataset",
        type=str.lower,
        required=True,
        choices=sorted(DATASET_SPECS),
        help="Dataset metadata mapping",
    )
    parser.add_argument(
        "--stage",
        choices=("preprocess", "split", "all"),
        default="all",
        help="Run preprocessing, splitting, or both stages (default: all)",
    )
    parser.add_argument("--preprocessed-output", type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--split-name", default="splits", help="Split directory name (default: splits)")
    parser.add_argument("--patient-col", help="Override the source patient column")
    parser.add_argument("--label-col", help="Override the source phenotype column")
    parser.add_argument("--cell-type-col", help="Override the source cell-type column")

    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-top-genes", type=int, default=3000)

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Patient-wise cell fraction; defaults to the dataset-specific setting",
    )
    parser.add_argument("--sample-seed", type=int, default=30)
    parser.add_argument(
        "--drop-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop adata.raw in split files; defaults to the dataset-specific setting",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2")
    if len(args.seeds) == 0:
        parser.error("--seeds must contain at least one value")
    if args.min_cells <= 0 or args.target_sum <= 0 or args.n_top_genes <= 0:
        parser.error("preprocessing parameters must be positive")
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    spec = DATASET_SPECS[args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = args.preprocessed_output or (
        args.output_dir / f"{args.dataset}_preprocessed.h5ad"
    )
    resolve_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    split_path = args.split_output or (args.output_dir / args.split_name)

    prepared_input = args.input
    if args.stage in {"preprocess", "all"}:
        prepared_input = preprocess_dataset(
            args.input,
            preprocessed_path,
            spec,
            args,
        )
    if args.stage in {"split", "all"}:
        create_patient_splits(prepared_input, split_path, spec, args)


if __name__ == "__main__":
    main()
