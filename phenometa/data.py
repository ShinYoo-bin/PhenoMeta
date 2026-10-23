from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class PatientBag:
    patient_id: str
    label: int
    raw_label: str
    embeddings: torch.Tensor
    original_features: Union[torch.Tensor, sp.spmatrix]
    obs_names: np.ndarray
    obs: pd.DataFrame
    source_split: str


def _matrix(adata, key, *, densify=True):
    if key == "X": value = adata.X
    elif key in adata.obsm: value = adata.obsm[key]
    elif key in adata.layers: value = adata.layers[key]
    else: raise KeyError(f"embedding key {key!r} not in X, obsm={list(adata.obsm)}, layers={list(adata.layers)}")
    if sp.issparse(value):
        if not densify:
            return sp.csr_matrix(value, dtype=np.float32)
        value = value.toarray()
    return np.asarray(value, dtype=np.float32, order="C")


def _matrix_rows(adata, key, rows):
    """Read selected rows without materializing the complete backed matrix."""
    if key == "X": value = adata.X
    elif key in adata.obsm: value = adata.obsm[key]
    elif key in adata.layers: value = adata.layers[key]
    else: raise KeyError(f"embedding key {key!r} not in X, obsm={list(adata.obsm)}, layers={list(adata.layers)}")
    rows = np.asarray(rows, dtype=np.int64)
    try:
        selected = value[rows]
    except TypeError:
        # Some backed sparse implementations require monotonically increasing
        # indices. Restore the requested order after reading sorted positions.
        order = np.argsort(rows)
        inverse = np.argsort(order)
        selected = value[rows[order]][inverse]
    if sp.issparse(selected):
        return sp.csr_matrix(selected, dtype=np.float32)
    return torch.from_numpy(np.asarray(selected, dtype=np.float32, order="C"))


def infer_label_map(adata, label_col):
    values = adata.obs[label_col]
    unique = sorted(values.dropna().astype(str).unique().tolist())
    return {label: i for i, label in enumerate(unique)}


def load_patient_bags(path, patient_col="patient", label_col="label", embedding_key="X_latent",
                      label_map: Optional[Dict[str, int]] = None, source_split="train",
                      original_feature_path=None, original_feature_key="X"):
    adata = ad.read_h5ad(path)
    for col in (patient_col, label_col):
        if col not in adata.obs: raise KeyError(f"missing obs[{col!r}] in {path}")
    label_map = label_map or infer_label_map(adata, label_col)
    matrix = _matrix(adata, embedding_key)
    original = None
    original_positions = None
    if original_feature_path is not None:
        original = ad.read_h5ad(original_feature_path, backed="r")
        if not original.obs_names.is_unique or not adata.obs_names.is_unique:
            raise ValueError("embedding and original h5ad require unique obs_names")
        original_positions = original.obs_names.get_indexer(adata.obs_names)
        if bool((original_positions < 0).any()):
            missing = adata.obs_names[original_positions < 0]
            raise KeyError(f"{len(missing)} embedded cells are absent from original input; first={missing[:5].tolist()}")
    bags = []
    grouped = adata.obs.groupby(patient_col, sort=True, observed=False).indices
    for patient, idx in grouped.items():
        idx = np.asarray(idx, dtype=np.int64)
        labels = adata.obs.iloc[idx][label_col].astype(str).unique()
        if len(labels) != 1: raise ValueError(f"patient={patient} has multiple labels: {labels.tolist()}")
        raw = labels[0]
        if raw not in label_map: raise KeyError(f"test label {raw!r} absent from training label map")
        if original is None:
            patient_original = torch.from_numpy(matrix[idx])
        else:
            # Keep only this patient's sparse rows in memory.  In particular,
            # never densify or load a complete multi-patient raw split.
            patient_original = _matrix_rows(
                original, original_feature_key, original_positions[idx])
        bags.append(PatientBag(str(patient), label_map[raw], raw, torch.from_numpy(matrix[idx]),
                               patient_original,
                               np.asarray(adata.obs_names[idx].astype(str)), adata.obs.iloc[idx].copy(), source_split))
    if original is not None and getattr(original, "file", None) is not None:
        original.file.close()
    return bags, label_map


def discover_repeats(input_dir):
    return sorted(int(p.name.split("_")[-1]) for p in Path(input_dir).glob("repeat_*") if p.is_dir())


def discover_folds(input_dir, repeat):
    root = Path(input_dir) / f"repeat_{repeat}"
    out = []
    for p in root.glob("fold_*_train.h5ad"):
        fold = int(p.name.split("_")[1])
        if (root / f"fold_{fold}_test.h5ad").exists(): out.append(fold)
    return sorted(out)


def split_train_validation(bags: List[PatientBag], val_fraction=0.25, seed=3407):
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    indices, labels = np.arange(len(bags)), np.asarray([b.label for b in bags])
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, labels))
    except ValueError:
        train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, shuffle=True)
    train = [bags[i] for i in train_idx]
    val = [bags[i] for i in val_idx]
    for bag in val: bag.source_split = "validation"
    return train, val
