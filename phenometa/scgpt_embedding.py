from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .patient_cache import (assemble, cache_complete, durable_write_h5ad, populate,
                            obs_metadata_matches, quarantine_corrupt, signature, valid_h5ad)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCGPT_SOURCE = Path(os.environ.get("SCGPT_SOURCE_DIR", PROJECT_ROOT / "external" / "scGPT")).expanduser()
DEFAULT_SCGPT_CHECKPOINT = Path(os.environ.get("SCGPT_CHECKPOINT_DIR", PROJECT_ROOT / "external" / "scGPT_checkpoint")).expanduser()


def _dense_float32(value):
    if sp.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32, order="C")


def _load_scgpt(source_dir):
    source_dir = Path(source_dir).resolve()
    if not (source_dir / "scgpt").is_dir():
        raise FileNotFoundError(f"scGPT source package not found: {source_dir / 'scgpt'}")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import scgpt
    return scgpt


def _vocab(checkpoint_dir):
    path = Path(checkpoint_dir) / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(f"scGPT vocab not found: {path}")
    with open(path, encoding="utf-8") as stream:
        return set(map(str, json.load(stream).keys()))


def _gene_names(adata, requested, vocab=None):
    adata.var["index"] = adata.var_names.astype(str)
    if "symbol" in adata.var:
        adata.var["symbol"] = adata.var["symbol"].astype(str).str.strip()
    if requested == "auto":
        preferred = ["symbol", "GeneSymbol", "gene_symbol", "gene_symbols", "var_names", "feature_name", "index"]
        candidates = [name for name in preferred if name == "index" or name in adata.var]
        if vocab is None:
            resolved = candidates[0]
        else:
            overlaps = {}
            for name in candidates:
                values = adata.var_names.astype(str) if name == "index" else adata.var[name].astype(str).str.strip()
                overlaps[name] = len(set(map(str, values)) & vocab)
            resolved = max(candidates, key=lambda name: overlaps[name])
            if overlaps[resolved] == 0:
                raise ValueError(f"No candidate gene column matched scGPT vocab; overlaps={overlaps}")
    else:
        resolved = requested
    if resolved == "index":
        names = adata.var_names.astype(str).to_numpy()
    elif resolved in adata.var:
        names = adata.var[resolved].astype(str).str.strip().to_numpy()
    else:
        raise KeyError(f"gene column {resolved!r} not found in adata.var")
    return resolved, np.char.strip(np.asarray(names, dtype=str))


def embed_split_h5ad(
    input_path,
    output_path,
    checkpoint_dir=DEFAULT_SCGPT_CHECKPOINT,
    source_dir=DEFAULT_SCGPT_SOURCE,
    patient_col="patient",
    gene_col="auto",
    batch_size=2000,
    max_length=1200,
    device="cuda",
    use_fast_transformer=False,
    overwrite=False,
    patient_cache_dir=None,
    log=None,
    repeat=-1,
    fold=-1,
    split="unknown",
):
    """Create and cache patient-wise scGPT embeddings for one fold split."""
    input_path, output_path = Path(input_path), Path(output_path)
    cache_signature = signature({"backend": "scgpt", "checkpoint": str(Path(checkpoint_dir).resolve()),
        "gene_col": gene_col, "max_length": int(max_length), "use_fast_transformer": bool(use_fast_transformer)})
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        if not valid_h5ad(output_path) or not obs_metadata_matches(input_path, output_path):
            quarantined = quarantine_corrupt(output_path)
            if log:
                log.warning(
                    "[scGPT cache] unreadable or metadata-mismatched file quarantined %s -> %s",
                    output_path, quarantined,
                )
        else:
            if patient_cache_dir and not cache_complete(input_path, patient_cache_dir, patient_col, cache_signature):
                populate(input_path, output_path, patient_cache_dir, patient_col, "X", cache_signature)
            if log: log.info("[scGPT cache] reuse %s", output_path)
            return output_path
    if patient_cache_dir and not overwrite and cache_complete(input_path, patient_cache_dir, patient_col, cache_signature):
        if log: log.info("[scGPT patient cache] assemble split=%s", input_path)
        return assemble(input_path, output_path, patient_cache_dir, patient_col, cache_signature)
    scgpt = _load_scgpt(source_dir)
    checkpoint_dir = Path(checkpoint_dir).resolve()
    for required in ("vocab.json", "args.json", "best_model.pt"):
        if not (checkpoint_dir / required).exists():
            raise FileNotFoundError(f"scGPT checkpoint file not found: {checkpoint_dir / required}")
    vocab = _vocab(checkpoint_dir)
    adata = ad.read_h5ad(input_path)
    if patient_col not in adata.obs:
        raise KeyError(f"obs[{patient_col!r}] not found in {input_path}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("scGPT input requires unique obs_names and var_names")
    resolved, names = _gene_names(adata, gene_col, vocab)
    gene_mask = np.isin(names, list(vocab))
    if not bool(gene_mask.any()):
        raise ValueError(f"No genes from {input_path} matched {checkpoint_dir / 'vocab.json'} using gene_col={resolved}")
    start = time.time(); embeddings, observations = [], []
    patient_values = adata.obs[patient_col].astype(str).to_numpy()
    patients = pd.unique(patient_values)
    if log:
        log.info("[scGPT] input=%s patients=%d matched_genes=%d/%d batch_size=%d", input_path,
                 len(patients), int(gene_mask.sum()), adata.n_vars, batch_size)
    for number, patient in enumerate(patients, 1):
        indices = np.flatnonzero(patient_values == patient)
        if log:
            log.info(
                "[patient:start] stage=scgpt repeat=%s fold=%s split=%s epoch=-1 "
                "patient_id=%s n_cells=%d n_metacells=na patient_index=%d/%d",
                repeat, fold, split, patient, len(indices), number, len(patients),
            )
        try:
            sub = adata[indices, gene_mask].copy()
            sub.var["index"] = sub.var_names.astype(str)
            if "symbol" in sub.var:
                sub.var["symbol"] = sub.var["symbol"].astype(str).str.strip()
            values = _dense_float32(sub.X)
            if not np.isfinite(values).all() or bool((values < 0).any()):
                raise ValueError(f"patient={patient}: scGPT input contains NaN/Inf or negative values")
            keep = values.sum(1) > 0
            if not bool(keep.any()):
                raise ValueError(f"patient={patient}: all cells are zero after scGPT vocab filtering")
            if not bool(keep.all()):
                sub = sub[keep].copy()
            embedded = scgpt.tasks.embed_data(
                sub,
                model_dir=checkpoint_dir,
                gene_col=resolved,
                batch_size=int(batch_size),
                max_length=int(max_length),
                device=device,
                use_fast_transformer=bool(use_fast_transformer),
                return_new_adata=False,
            )
            if "X_scGPT" not in embedded.obsm:
                raise KeyError("scgpt.tasks.embed_data did not produce obsm['X_scGPT']")
            matrix = np.asarray(embedded.obsm["X_scGPT"], dtype=np.float32)
            if matrix.shape[0] != sub.n_obs or not np.isfinite(matrix).all():
                raise ValueError(f"patient={patient}: invalid scGPT embedding shape/values")
            embeddings.append(matrix); observations.append(sub.obs.copy())
            if log:
                log.info("[patient:done] stage=scgpt repeat=%s fold=%s split=%s "
                         "patient_id=%s n_cells=%d patient_index=%d/%d",
                         repeat, fold, split, patient, sub.n_obs, number, len(patients))
        except BaseException as exc:
            if log:
                log.exception(
                    "[patient:failed] stage=scgpt repeat=%s fold=%s split=%s epoch=-1 "
                    "patient_id=%s n_cells=%d n_metacells=na exception_type=%s exception=%s",
                    repeat, fold, split, patient, len(indices), type(exc).__name__, exc,
                )
            raise
    matrix = np.concatenate(embeddings, axis=0)
    obs = pd.concat(observations, axis=0)
    var = pd.DataFrame(index=[f"scGPT_{i}" for i in range(matrix.shape[1])])
    output = ad.AnnData(X=matrix, obs=obs, var=var)
    output.uns["embedding_method"] = "scGPT"
    output.uns["source_h5ad"] = str(input_path.resolve())
    output.uns["checkpoint_dir"] = str(checkpoint_dir)
    output.uns["embedding_batch_size"] = int(batch_size)
    durable_write_h5ad(output, output_path)
    if not obs_metadata_matches(input_path, output_path):
        quarantined = quarantine_corrupt(output_path)
        raise ValueError(
            f"new scGPT cache obs metadata differs from source; quarantined at {quarantined}"
        )
    if patient_cache_dir:
        populate(input_path, output_path, patient_cache_dir, patient_col, "X", cache_signature)
    info = {"input_path": str(input_path.resolve()), "output_path": str(output_path.resolve()),
            "checkpoint_dir": str(checkpoint_dir), "batch_size": int(batch_size), "n_cells": output.n_obs,
            "embedding_dim": output.n_vars, "elapsed_sec": time.time() - start}
    with open(output_path.with_suffix(".json"), "w") as stream:
        json.dump(info, stream, indent=2)
    if log: log.info("[scGPT] wrote %s shape=%s elapsed=%.1fs", output_path, output.shape, info["elapsed_sec"])
    return output_path
