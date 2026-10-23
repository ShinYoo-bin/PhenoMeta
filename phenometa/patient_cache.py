from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _name(patient) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patient)).strip("._") or "patient"
    digest = hashlib.sha1(str(patient).encode()).hexdigest()[:10]
    return f"{safe}_{digest}.h5ad"


def _matrix(adata, key):
    if key == "X": value = adata.X
    elif key in adata.obsm: value = adata.obsm[key]
    elif key in adata.layers: value = adata.layers[key]
    else: raise KeyError(f"missing embedding key={key!r}")
    if sp.issparse(value): value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def cell_ids(adata, indices=None) -> pd.Index:
    """Return stable source-cell IDs, falling back to the AnnData index.

    The reduction split writer resets ``adata.obs_names`` to 0..N for every
    file, while preserving the real barcode/source ID in ``obs['obs_names']``.
    Patient caches must therefore use the preserved column across folds.
    """
    if "obs_names" in adata.obs:
        values = pd.Index(adata.obs["obs_names"].astype(str), dtype="object")
        if values.notna().all() and values.is_unique:
            return values if indices is None else values[np.asarray(indices, dtype=np.int64)]
    values = pd.Index(adata.obs_names.astype(str), dtype="object")
    return values if indices is None else values[np.asarray(indices, dtype=np.int64)]


def positions_for_ids(adata, expected) -> np.ndarray:
    available = cell_ids(adata)
    positions = available.get_indexer(pd.Index(expected, dtype="object"))
    if bool((positions < 0).any()):
        raise KeyError(f"cache is missing {int((positions < 0).sum())} source cell(s)")
    return positions.astype(np.int64)


def _patients(raw_path, patient_col):
    raw = ad.read_h5ad(raw_path, backed="r")
    if patient_col not in raw.obs: raise KeyError(f"missing obs[{patient_col!r}] in {raw_path}")
    return raw, raw.obs.groupby(patient_col, sort=True, observed=False).indices


def valid_h5ad(path) -> bool:
    """Treat partial/truncated cache files as cache misses."""
    try:
        item = ad.read_h5ad(path, backed="r")
        _ = (item.n_obs, item.n_vars, item.obs_names[:1])
        if getattr(item, "file", None) is not None: item.file.close()
        return True
    except (OSError, ValueError, KeyError):
        return False


def _normalized_obs_values(series) -> np.ndarray:
    """Return comparison-safe observation metadata, including missing values."""
    return series.astype("string").fillna("<NA>").to_numpy(dtype=str)


def obs_metadata_matches(source_path, candidate_path, columns=None) -> bool:
    """Check that a cache preserves source obs metadata for every stable cell ID.

    Cache rows may legitimately be reordered, so metadata is aligned with the
    stable source-cell IDs returned by :func:`cell_ids` before comparison.
    """
    source = candidate = None
    try:
        source = ad.read_h5ad(source_path, backed="r")
        candidate = ad.read_h5ad(candidate_path, backed="r")
        source_ids = cell_ids(source)
        candidate_ids = cell_ids(candidate)
        if (len(source_ids) != len(candidate_ids) or not source_ids.is_unique
                or not candidate_ids.is_unique or set(source_ids) != set(candidate_ids)):
            return False
        columns = list(source.obs.columns) if columns is None else list(columns)
        if any(column not in source.obs or column not in candidate.obs for column in columns):
            return False
        source_positions = source_ids.get_indexer(candidate_ids)
        if bool((source_positions < 0).any()):
            return False
        for column in columns:
            expected = _normalized_obs_values(source.obs[column])[source_positions]
            observed = _normalized_obs_values(candidate.obs[column])
            if not np.array_equal(expected, observed):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
    finally:
        for item in (source, candidate):
            if item is not None and getattr(item, "file", None) is not None:
                item.file.close()


def quarantine_corrupt(path) -> Path:
    path = Path(path)
    destination = path.with_name(f"{path.name}.corrupt.{time.time_ns()}")
    path.replace(destination)
    return destination


def durable_write_h5ad(adata, path):
    """Validate and fsync a temporary h5ad before publishing it atomically."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.h5ad")
    try:
        adata.write_h5ad(temporary)
        if not valid_h5ad(temporary):
            raise OSError(f"new h5ad cache failed validation: {temporary}")
        with temporary.open("rb") as stream: os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if temporary.exists(): temporary.unlink()


def cache_complete(raw_path, cache_dir, patient_col, cache_signature):
    raw, groups = _patients(raw_path, patient_col); cache_dir = Path(cache_dir)
    try:
        for patient, indices in groups.items():
            path = cache_dir / _name(patient)
            if not path.is_file(): return False
            if not valid_h5ad(path): return False
            item = ad.read_h5ad(path, backed="r")
            expected = cell_ids(raw, indices)
            observed = cell_ids(item)
            matches = (item.uns.get("cache_signature") == cache_signature
                       and len(observed) == len(expected)
                       and set(observed) == set(expected)
                       and patient_col in item.obs
                       and bool((_normalized_obs_values(item.obs[patient_col]) == str(patient)).all()))
            if getattr(item, "file", None) is not None: item.file.close()
            if not matches: return False
        return True
    finally:
        if getattr(raw, "file", None) is not None: raw.file.close()


def populate(raw_path, embedded_path, cache_dir, patient_col, embedding_key, cache_signature):
    if not obs_metadata_matches(raw_path, embedded_path):
        raise ValueError(f"embedded split obs metadata differs from source: {embedded_path}")
    raw = ad.read_h5ad(raw_path); embedded = ad.read_h5ad(embedded_path)
    if not raw.obs_names.is_unique or not embedded.obs_names.is_unique: raise ValueError("patient cache requires unique obs_names")
    missing = raw.obs_names.difference(embedded.obs_names)
    if len(missing): raise ValueError(f"embedded split misses {len(missing)} source cells")
    embedded = embedded[raw.obs_names]
    values = _matrix(embedded, embedding_key); cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    for patient, indices in raw.obs.groupby(patient_col, sort=True, observed=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64); path = cache_dir / _name(patient)
        out = ad.AnnData(X=values[indices], obs=raw.obs.iloc[indices].copy())
        out.obs_names = cell_ids(raw, indices)
        out.uns.update(cache_signature=cache_signature, patient_id=str(patient), embedding_key="X")
        durable_write_h5ad(out, path)


def assemble(raw_path, output_path, cache_dir, patient_col, cache_signature, output_key="X"):
    raw = ad.read_h5ad(raw_path); cache_dir = Path(cache_dir)
    matrix = None
    for patient, indices in raw.obs.groupby(patient_col, sort=True, observed=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64); item = ad.read_h5ad(cache_dir / _name(patient))
        expected = cell_ids(raw, indices)
        if item.uns.get("cache_signature") != cache_signature or set(cell_ids(item)) != set(expected):
            raise ValueError(f"invalid patient cache for patient={patient}")
        if patient_col not in item.obs or not bool(
                (_normalized_obs_values(item.obs[patient_col]) == str(patient)).all()):
            raise ValueError(f"patient metadata mismatch in cache for patient={patient}")
        positions = positions_for_ids(item, expected)
        values = np.asarray(item.X[positions], dtype=np.float32)
        if matrix is None: matrix = np.empty((raw.n_obs, values.shape[1]), dtype=np.float32)
        matrix[indices] = values
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError(f"assembled cache has missing/invalid embeddings: {raw_path}")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    out = ad.AnnData(X=matrix, obs=raw.obs.copy(deep=True),
                     var=pd.DataFrame(index=[f"embedding_{i}" for i in range(matrix.shape[1])]))
    if output_key != "X": out.obsm[output_key] = matrix
    out.uns.update(cache_signature=cache_signature, assembled_from_patient_cache=True)
    durable_write_h5ad(out, output_path)
    if not obs_metadata_matches(raw_path, output_path):
        quarantined = quarantine_corrupt(output_path)
        raise ValueError(
            f"assembled cache obs metadata differs from source; quarantined at {quarantined}"
        )
    return output_path
