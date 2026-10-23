from __future__ import annotations

import json
import math
import re
import zlib
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .patient_cache import (cell_ids, durable_write_h5ad, obs_metadata_matches,
                            positions_for_ids, quarantine_corrupt, valid_h5ad)
from .utils import atomic_write_frame


@dataclass
class GraphAEConfig:
    implementation_version: int = 2
    latent_dim: int = 32
    knn_k: int = 5
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-2
    lambda_rec: float = 0.6
    lambda_lap: float = 0.2
    lambda_con: float = 0.2
    temperature: float = 0.2
    negatives_per_anchor: int = 32
    max_anchors: int = 1024
    early_stopping_min_delta: float = 1e-5
    early_stopping_patience: int = 10


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def _matrix(adata, key):
    if key == "X":
        value = adata.X
    elif key in adata.obsm:
        value = adata.obsm[key]
    elif key in adata.layers:
        value = adata.layers[key]
    else:
        raise KeyError(f"embedding key {key!r} absent from {adata.filename}")
    if sp.issparse(value):
        value = value.toarray()
    value = np.asarray(value, dtype=np.float32, order="C")
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid AE input shape/values: {value.shape}")
    return value


def _graph_matrix(adata, key):
    if key == "X":
        value = adata.X
    elif key in adata.obsm:
        value = adata.obsm[key]
    elif key in adata.layers:
        value = adata.layers[key]
    else:
        raise KeyError(f"graph key {key!r} absent from {adata.filename}")
    if sp.issparse(value):
        value = value.tocsr().astype(np.float32)
        if not np.isfinite(value.data).all():
            raise ValueError("graph input contains NaN/Inf")
        return value
    value = np.asarray(value, dtype=np.float32, order="C")
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid graph input shape/values: {value.shape}")
    return value


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _patient_cache_name(value):
    return f"{_safe_name(value)}_{hashlib.sha1(str(value).encode()).hexdigest()[:10]}.h5ad"


def _patient_seed(seed, patient):
    return (int(seed) + zlib.crc32(str(patient).encode("utf-8"))) % (2**31 - 1)


def _matching_legacy_config(cached, cfg):
    """Accept completed pre-patient-cache split artifacts without retraining."""
    raw = cached.uns.get("graph_ae_config_json")
    try:
        saved = json.loads(raw) if isinstance(raw, str) else dict(cached.uns.get("graph_ae_config", {}))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return all(saved.get(key) == value for key, value in asdict(cfg).items())


def _populate_patient_cache_from_split(cached, patient_col, patient_cache_dir, config_json):
    """Migrate a compatible completed split latent file into patient artifacts."""
    if not patient_cache_dir or patient_col not in cached.obs:
        return
    patient_cache_dir = Path(patient_cache_dir)
    patient_cache_dir.mkdir(parents=True, exist_ok=True)
    latent = np.asarray(cached.obsm["X_latent"], dtype=np.float32)
    for patient, indices in cached.obs.groupby(patient_col, sort=True, observed=False).indices.items():
        cache_path = patient_cache_dir / _patient_cache_name(patient)
        if cache_path.is_file():
            continue
        indices = np.asarray(indices, dtype=np.int64)
        patient_latent = latent[indices]
        artifact = ad.AnnData(X=patient_latent, obs=cached.obs.iloc[indices].copy())
        artifact.obs_names = cell_ids(cached, indices)
        artifact.obsm["X_latent"] = patient_latent
        artifact.uns.update(graph_ae_config_json=config_json, patient_id=str(patient), epochs_trained=0)
        durable_write_h5ad(artifact, cache_path)


def build_patient_knn(x, k):
    if x.shape[0] < 2:
        return np.empty((x.shape[0], 0), dtype=np.int64)
    effective_k = min(int(k), x.shape[0] - 1)
    search = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    search.fit(x)
    return search.kneighbors(x, return_distance=False)[:, 1:].astype(np.int64)


def cosine_reconstruction_loss(x, reconstructed):
    return (1.0 - F.cosine_similarity(x, reconstructed, dim=1, eps=1e-8)).mean()


def laplacian_loss(z, knn):
    if knn.shape[1] == 0:
        return z.new_zeros(())
    return ((z[:, None, :] - z[knn]) ** 2).sum(2).mean()


def contrastive_loss(z, knn, temperature, negatives_per_anchor, max_anchors):
    n = z.shape[0]
    if n < 2 or knn.shape[1] == 0:
        return z.new_zeros(())
    anchors = torch.arange(n, device=z.device)
    if anchors.numel() > max_anchors:
        anchors = anchors[torch.randperm(n, device=z.device)[:max_anchors]]
    positive_col = torch.randint(0, knn.shape[1], (anchors.numel(),), device=z.device)
    positives = knn[anchors, positive_col]
    negatives = torch.randint(0, n, (anchors.numel(), negatives_per_anchor), device=z.device)
    invalid = (negatives == anchors[:, None]) | (negatives == positives[:, None])
    normalized = F.normalize(z, p=2, dim=1, eps=1e-8)
    positive_score = (normalized[anchors] * normalized[positives]).sum(1, keepdim=True) / temperature
    negative_score = (normalized[anchors, None, :] * normalized[negatives]).sum(2) / temperature
    negative_score = negative_score.masked_fill(invalid, -1e9)
    logits = torch.cat((positive_score, negative_score), dim=1)
    return F.cross_entropy(logits, torch.zeros(anchors.numel(), dtype=torch.long, device=z.device))


def fit_patient(x_np, knn_x_np, patient, cfg, device, seed, artifact_dir, log=None):
    if x_np.shape[0] < 2:
        raise ValueError(f"patient={patient} has fewer than two cells; graph AE cannot train")
    local_seed = _patient_seed(seed, patient)
    np.random.seed(local_seed); torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    knn_np = build_patient_knn(knn_x_np, cfg.knn_k)
    knn = torch.as_tensor(knn_np, dtype=torch.long, device=device)
    model = GraphAutoEncoder(x.shape[1], cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, stale, history = math.inf, 0, []
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_columns = ["epoch", "total_loss", "rec_loss", "lap_loss", "con_loss"]
    atomic_write_frame(pd.DataFrame(columns=history_columns), artifact_dir / "loss_history.csv")
    for epoch in range(1, cfg.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        z, reconstructed = model(x)
        rec = cosine_reconstruction_loss(x, reconstructed)
        lap = laplacian_loss(z, knn)
        con = contrastive_loss(z, knn, cfg.temperature, cfg.negatives_per_anchor, cfg.max_anchors)
        total = cfg.lambda_rec * rec + cfg.lambda_lap * lap + cfg.lambda_con * con
        total.backward(); optimizer.step()
        row = {"epoch": epoch, "total_loss": float(total.detach()), "rec_loss": float(rec.detach()),
               "lap_loss": float(lap.detach()), "con_loss": float(con.detach())}
        history.append(row)
        atomic_write_frame(pd.DataFrame(history, columns=history_columns), artifact_dir / "loss_history.csv")
        improvement = best - row["total_loss"]
        if improvement >= cfg.early_stopping_min_delta:
            best, stale = row["total_loss"], 0
        else:
            stale += 1
            if stale >= cfg.early_stopping_patience:
                break
    model.eval()
    with torch.no_grad():
        latent = model.encoder(x).cpu().numpy().astype(np.float32)
    np.save(artifact_dir / "knn_indices.npy", knn_np)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    atomic_write_frame(pd.DataFrame(history, columns=history_columns), artifact_dir / "loss_history.csv")
    (artifact_dir / "config.json").write_text(json.dumps({**asdict(cfg), "patient": str(patient),
        "n_cells": int(x_np.shape[0]), "input_dim": int(x_np.shape[1]),
        "knn_input_dim": int(knn_x_np.shape[1]), "epochs_trained": len(history)}, indent=2))
    if log:
        log.info("[graph-ae] patient=%s cells=%d input_dim=%d latent_dim=%d epochs=%d loss=%.6f",
                 patient, x_np.shape[0], x_np.shape[1], cfg.latent_dim, len(history), history[-1]["total_loss"])
    return latent, len(history)


def embed_split_graph_ae(input_path, output_path, *, patient_col, embedding_key, cfg, device,
                         seed, graph_input_path=None, graph_input_key="X", overwrite=False,
                         patient_cache_dir=None, log=None, repeat=-1, fold=-1, split="unknown"):
    input_path, output_path = Path(input_path), Path(output_path)
    config_json = json.dumps({"cache_version": 2, "patient_seed_policy": "fold_independent", **asdict(cfg)}, sort_keys=True)
    if output_path.exists() and not overwrite:
        valid = False
        cached = source = None
        try:
            if valid_h5ad(output_path):
                cached = ad.read_h5ad(output_path, backed="r")
                source = ad.read_h5ad(input_path, backed="r")
                shape_and_cells_match = ("X_latent" in cached.obsm
                                         and cached.obsm["X_latent"].shape[1] == cfg.latent_dim
                                         and cached.obs_names.equals(source.obs_names))
                current_cache = cached.uns.get("graph_ae_config_json") == config_json
                legacy_cache = _matching_legacy_config(cached, cfg)
                valid = (shape_and_cells_match and (current_cache or legacy_cache)
                         and obs_metadata_matches(input_path, output_path))
                if valid:
                    _populate_patient_cache_from_split(cached, patient_col, patient_cache_dir, config_json)
                    if log:
                        kind = "cache" if current_cache else "compatible legacy split cache"
                        log.info("[graph-ae] using %s=%s", kind, output_path)
        finally:
            for item in (cached, source):
                if item is not None and getattr(item, "file", None) is not None:
                    item.file.close()
        if valid:
            return output_path
        quarantined = quarantine_corrupt(output_path)
        if log:
            log.warning(
                "[graph-ae cache] incompatible or metadata-mismatched file quarantined %s -> %s",
                output_path, quarantined,
            )
    adata = ad.read_h5ad(input_path)
    if patient_col not in adata.obs:
        raise KeyError(f"missing obs[{patient_col!r}] in {input_path}")
    matrix = _matrix(adata, embedding_key)
    if graph_input_path is None:
        graph_matrix = matrix
    else:
        graph_adata = ad.read_h5ad(graph_input_path)
        if not graph_adata.obs_names.is_unique or not adata.obs_names.is_unique:
            raise ValueError("AE embedding and graph inputs require unique obs_names")
        missing = adata.obs_names.difference(graph_adata.obs_names)
        if len(missing):
            raise KeyError(f"{len(missing)} AE cells are absent from graph input")
        graph_matrix = _graph_matrix(graph_adata[adata.obs_names], graph_input_key)
    latent = np.empty((adata.n_obs, cfg.latent_dim), dtype=np.float32)
    summaries = []
    artifacts = output_path.parent / f"{output_path.stem}_patients"
    patient_cache_dir = Path(patient_cache_dir) if patient_cache_dir else None
    if patient_cache_dir: patient_cache_dir.mkdir(parents=True, exist_ok=True)
    for patient, indices in adata.obs.groupby(patient_col, sort=True, observed=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        if log:
            log.info(
                "[patient:start] stage=graph_ae repeat=%s fold=%s split=%s epoch=-1 "
                "patient_id=%s n_cells=%d n_metacells=na",
                repeat, fold, split, patient, len(indices),
            )
        try:
            cache_path = patient_cache_dir / _patient_cache_name(patient) if patient_cache_dir else None
            expected_names = cell_ids(adata, indices)
            if cache_path and cache_path.is_file() and not overwrite:
                valid = valid_h5ad(cache_path)
                cached_patient = ad.read_h5ad(cache_path) if valid else None
                valid = (valid and cached_patient.uns.get("graph_ae_config_json") == config_json and
                         set(cell_ids(cached_patient)) == set(expected_names) and "X_latent" in cached_patient.obsm)
            else:
                valid = False
            if valid:
                positions = positions_for_ids(cached_patient, expected_names)
                patient_latent = np.asarray(cached_patient.obsm["X_latent"][positions], dtype=np.float32)
                epochs = int(cached_patient.uns.get("epochs_trained", 0))
                if log:
                    log.info("[graph-ae patient cache] patient=%s cells=%d", patient, len(indices))
            else:
                patient_latent, epochs = fit_patient(
                    matrix[indices], graph_matrix[indices], patient, cfg, device, seed,
                    artifacts / _safe_name(patient), log)
                if cache_path:
                    cached_patient = ad.AnnData(X=patient_latent, obs=adata.obs.iloc[indices].copy())
                    cached_patient.obs_names = expected_names
                    cached_patient.obsm["X_latent"] = patient_latent
                    cached_patient.uns.update(graph_ae_config_json=config_json, patient_id=str(patient),
                                              epochs_trained=int(epochs))
                    durable_write_h5ad(cached_patient, cache_path)
            latent[indices] = patient_latent
            summaries.append({"patient": str(patient), "n_cells": len(indices), "epochs_trained": epochs})
            if log:
                log.info("[patient:done] stage=graph_ae repeat=%s fold=%s split=%s "
                         "patient_id=%s n_cells=%d epochs=%d",
                         repeat, fold, split, patient, len(indices), epochs)
        except BaseException as exc:
            if log:
                log.exception(
                    "[patient:failed] stage=graph_ae repeat=%s fold=%s split=%s epoch=-1 "
                    "patient_id=%s n_cells=%d n_metacells=na exception_type=%s exception=%s",
                    repeat, fold, split, patient, len(indices), type(exc).__name__, exc,
                )
            raise
    if not np.isfinite(latent).all():
        raise ValueError("graph AE produced NaN/Inf latent values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = ad.AnnData(X=latent, obs=adata.obs.copy())
    output.obs_names = adata.obs_names.copy()
    output.obsm["X_latent"] = latent
    output.uns["graph_ae_config"] = asdict(cfg)
    output.uns["graph_ae_config_json"] = config_json
    durable_write_h5ad(output, output_path)
    if not obs_metadata_matches(input_path, output_path):
        quarantined = quarantine_corrupt(output_path)
        raise ValueError(
            f"new graph-AE cache obs metadata differs from source; quarantined at {quarantined}"
        )
    pd.DataFrame(summaries).to_csv(output_path.parent / f"{output_path.stem}_summary.csv", index=False)
    return output_path
