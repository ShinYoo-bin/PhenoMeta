from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def row_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class CellPrototypeAssignmentGate(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, interactions: bool = False):
        super().__init__()
        self.interactions = interactions
        pair_dim = dim * (4 if interactions else 2)
        self.v_branch = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.ELU())
        self.u_branch = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.Sigmoid())
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, cells: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if cells.shape != prototypes.shape:
            raise ValueError("cells and prototypes must have matching [N, K, D] shapes")
        pieces = [cells, prototypes]
        if self.interactions:
            pieces.extend([cells * prototypes, torch.abs(cells - prototypes)])
        pair = torch.cat(pieces, dim=-1)
        return self.output_layer(self.v_branch(pair) * self.u_branch(pair)).squeeze(-1)


def approximate_fps(x: torch.Tensor, k: int, subset_size: int, seed: int) -> torch.Tensor:
    """Cosine FPS on a reproducible random subset; returns copied prototypes."""
    n = x.shape[0]
    k = min(max(1, k), n)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    subset_idx = torch.randperm(n, generator=gen)[: min(n, subset_size)].to(x.device)
    subset = row_normalize(x[subset_idx])
    selected = [0]
    min_dist = 1.0 - subset @ subset[0]
    min_dist[0] = -1
    for _ in range(1, min(k, subset.shape[0])):
        nxt = int(min_dist.argmax())
        selected.append(nxt)
        min_dist = torch.minimum(min_dist, 1.0 - subset @ subset[nxt])
        min_dist[nxt] = -1
    return x[subset_idx[torch.tensor(selected, device=x.device)]].clone()


def cosine_topk(x: torch.Tensor, p: torch.Tensor, k: int, chunk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xn, pn = row_normalize(x), row_normalize(p)
    indices, similarities = [], []
    for start in range(0, x.shape[0], chunk):
        sim = xn[start : start + chunk] @ pn.T
        vals, idx = torch.topk(sim, min(k, p.shape[0]), dim=1)
        indices.append(idx)
        similarities.append(vals)
    return torch.cat(indices), torch.cat(similarities)


def _candidate_bias(x, p, indices, gate):
    cells = x[:, None, :].expand(-1, indices.shape[1], -1)
    protos = p[indices]
    return gate(cells, protos)


def compute_refinement_soft_assignment(
    x: torch.Tensor,
    p: torch.Tensor,
    usage: torch.Tensor,
    gate: CellPrototypeAssignmentGate,
    cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """Ordinary softmax assignment used only for EM prototype updates."""
    k = min(max(cfg.min_top_k, int(p.shape[0] * cfg.top_k_ratio)), cfg.max_top_k, p.shape[0])
    indices, cosine = cosine_topk(x, p, k, cfg.search_chunk_size)
    usage_norm = usage / usage.mean().clamp_min(cfg.eps)
    base = cosine / (1.0 + cfg.usage_alpha * usage_norm[indices])
    if cfg.use_assignment_gate and cfg.use_bias_in_refinement:
        bias = _candidate_bias(x, p, indices, gate)
    else:
        bias = torch.zeros_like(base)
    logits = base + cfg.bias_weight * bias
    weights = torch.softmax(logits / cfg.refinement_tau, dim=1)
    return {"indices": indices, "cosine": cosine, "base": base, "bias": bias, "logits": logits, "weights": weights}


def _soft_update(x, indices, weights, n_prototypes, eps):
    d = x.shape[1]
    accum = torch.zeros(n_prototypes, d, device=x.device, dtype=x.dtype)
    mass = torch.zeros(n_prototypes, device=x.device, dtype=x.dtype)
    flat_idx = indices.reshape(-1)
    flat_w = weights.reshape(-1)
    accum.index_add_(0, flat_idx, (x[:, None, :] * weights[..., None]).reshape(-1, d))
    mass.index_add_(0, flat_idx, flat_w)
    return accum / mass[:, None].clamp_min(eps), mass


def _merge_once(p, usage, cfg):
    if p.shape[0] <= 1:
        return p, 0
    sim = row_normalize(p) @ row_normalize(p).T
    sim.fill_diagonal_(-1)
    pairs, used = [], set()
    flat = torch.argsort(sim.flatten(), descending=True)
    for value in flat.tolist():
        i, j = divmod(value, p.shape[0])
        if sim[i, j] < cfg.merge_threshold:
            break
        if i in used or j in used:
            continue
        ratio = max(float(usage[i]), float(usage[j])) / (min(float(usage[i]), float(usage[j])) + cfg.eps)
        if ratio <= cfg.merge_usage_ratio_threshold:
            pairs.append((i, j)); used.update((i, j))
    if not pairs:
        return p, 0
    mate = {a: b for a, b in pairs} | {b: a for a, b in pairs}
    rows, visited = [], set()
    for i in range(p.shape[0]):
        if i in visited:
            continue
        if i in mate:
            j = mate[i]
            rows.append((usage[i] * p[i] + usage[j] * p[j]) / (usage[i] + usage[j] + cfg.eps))
            visited.update((i, j))
        else:
            rows.append(p[i]); visited.add(i)
    return torch.stack(rows), len(pairs)


class PrototypeRefiner(nn.Module):
    """Detached EM-like outer operation. No ST assignment is used here."""
    def __init__(self, cfg: ModelConfig):
        super().__init__(); self.cfg = cfg

    @torch.no_grad()
    def forward(self, x, gate, seed=42):
        cfg, n = self.cfg, x.shape[0]
        k0 = min(max(int(cfg.init_scale * math.sqrt(n)), cfg.min_init_prototypes), cfg.max_init_prototypes, n)
        p = approximate_fps(x, k0, cfg.fps_subset_size, seed)
        removed_total = merged_total = rounds = 0
        last = None
        logs = []
        for outer in range(cfg.max_outer_iter):
            usage = torch.ones(p.shape[0], device=x.device, dtype=x.dtype)
            prev_loss = None
            for iteration in range(cfg.max_refinement_iter):
                assignment = compute_refinement_soft_assignment(x, p, usage, gate, cfg)
                p_new, usage_new = _soft_update(x, assignment["indices"], assignment["weights"], p.shape[0], cfg.eps)
                alive = usage_new > 0
                p_new[~alive] = p[~alive]
                loss = float((assignment["weights"] * (1 - assignment["cosine"])).sum())
                change = float((1 - F.cosine_similarity(p, p_new)).mean())
                entropy = float((-(assignment["weights"] * assignment["weights"].clamp_min(cfg.eps).log()).sum(1)).mean())
                rounds += 1; p, usage, last = p_new, usage_new, assignment
                rel = None if prev_loss is None else abs(loss - prev_loss) / (abs(prev_loss) + cfg.eps)
                logs.append({"outer_iter": outer, "refinement_iter": iteration, "objective": loss,
                             "relative_change": rel, "prototype_cosine_change": change,
                             "assignment_entropy": entropy, "n_prototypes": p.shape[0]})
                if rel is not None and rel < cfg.convergence_tol:
                    break
                prev_loss = loss
            alive = usage >= cfg.dead_eps * n
            if not bool(alive.any()): alive[usage.argmax()] = True
            removed = int((~alive).sum()); removed_total += removed
            p, usage = p[alive], usage[alive]
            p, merged = _merge_once(p, usage, cfg); merged_total += merged
            if removed == 0 and merged == 0:
                break
        # Recompute usage for the final prototype indexing.
        usage = torch.ones(p.shape[0], device=x.device, dtype=x.dtype)
        last = compute_refinement_soft_assignment(x, p, usage, gate, cfg)
        _, usage = _soft_update(x, last["indices"], last["weights"], p.shape[0], cfg.eps)
        return p.detach(), usage.detach(), {"initial_prototypes": k0, "final_prototypes": p.shape[0],
            "refinement_rounds": rounds, "removed": removed_total, "merged": merged_total, "iterations": logs}


def compute_final_st_hard_assignment(x, p, usage, gate, cfg, tau, training):
    k = min(max(cfg.min_top_k, int(p.shape[0] * cfg.top_k_ratio)), cfg.max_top_k, p.shape[0])
    indices, cosine = cosine_topk(x, p, k, cfg.search_chunk_size)
    usage_norm = usage / usage.mean().clamp_min(cfg.eps)
    base = cosine / (1.0 + cfg.usage_alpha * usage_norm[indices])
    bias = _candidate_bias(x, p, indices, gate) if cfg.use_assignment_gate and cfg.use_bias_in_final_assignment else torch.zeros_like(base)
    logits = base + cfg.bias_weight * bias
    scaled_logits = logits / cfg.refinement_tau
    if training and cfg.use_st_gumbel_final_assignment:
        candidate_hard = F.gumbel_softmax(scaled_logits, tau=tau, hard=True, dim=1)
    else:
        candidate_hard = F.one_hot(logits.argmax(1), logits.shape[1]).to(logits.dtype)
    full = torch.zeros(x.shape[0], p.shape[0], device=x.device, dtype=x.dtype)
    full.scatter_(1, indices, candidate_hard)
    return full, {"indices": indices, "base": base, "bias": bias, "logits": logits, "gumbel_logits": scaled_logits}


class SizeAwareAttentionPredictor(nn.Module):
    """Encode metacells, then pool abundance modulated by learned relevance."""
    def __init__(self, input_dim, n_classes, cfg):
        super().__init__()
        if cfg.pooling not in {"size_aware_attention", "size_weighted", "mean"}:
            raise ValueError(f"unsupported pooling: {cfg.pooling}")
        if cfg.pooling_hidden_dim <= 0:
            raise ValueError("pooling_hidden_dim must be positive")
        self.pooling = cfg.pooling
        self.eps = cfg.eps
        e1, e2 = cfg.predictor_encoder_dims
        c1, c2 = cfg.predictor_classifier_dims
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, e1), nn.ELU(), nn.Dropout(cfg.predictor_encoder_dropout),
            nn.Linear(e1, e2), nn.ELU(), nn.Dropout(cfg.predictor_encoder_dropout),
            nn.Linear(e2, cfg.predictor_hidden_dim),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(cfg.predictor_hidden_dim, cfg.pooling_hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.pooling_hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.predictor_hidden_dim, c1), nn.ELU(), nn.Dropout(cfg.classifier_dropout),
            nn.Linear(c1, c2), nn.ELU(), nn.Dropout(cfg.classifier_dropout),
            nn.Linear(c2, n_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
        # score=0 gives softmax(log(size_weight)) == size_weight, so the
        # Start from size-weighted pooling before learning relevance residuals.
        nn.init.zeros_(self.attention_score[-1].weight)

    def forward(self, metacells, counts):
        active = counts > 0
        if not bool(active.any()): raise ValueError("patient has no active metacell")
        encoded = self.instance_encoder(metacells[active])
        active_counts = counts[active].to(dtype=encoded.dtype)
        size_weights = active_counts / active_counts.sum().clamp_min(self.eps)
        scores = self.attention_score(encoded).squeeze(-1)
        if self.pooling == "mean":
            weights = encoded.new_full((encoded.shape[0],), 1.0 / encoded.shape[0])
        elif self.pooling == "size_weighted":
            weights = size_weights
        else:
            # softmax is invariant to the shared -log(total_count) term, so
            # log(count) is the stable equivalent of log(size_weight).
            weights = torch.softmax(active_counts.clamp_min(self.eps).log() + scores, dim=0)
        patient = (encoded * weights[:, None]).sum(0)
        return self.classifier(patient), patient, weights, size_weights, scores


class PhenoMetaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__(); self.cfg = cfg
        self.assignment_gate = CellPrototypeAssignmentGate(cfg.latent_dim, cfg.gate_hidden_dim, cfg.use_interaction_pair_features)
        self.refiner = PrototypeRefiner(cfg)
        feature_dim = cfg.metacell_feature_dim or cfg.latent_dim
        self.predictor = SizeAwareAttentionPredictor(feature_dim, cfg.n_classes, cfg)

    def reset_predictor(self):
        """Replace only the PP predictor while preserving the learned assignment gate."""
        parameter = next(self.assignment_gate.parameters())
        feature_dim = self.cfg.metacell_feature_dim or self.cfg.latent_dim
        self.predictor = SizeAwareAttentionPredictor(
            feature_dim, self.cfg.n_classes, self.cfg
        ).to(device=parameter.device, dtype=parameter.dtype)

    def forward(self, cell_embeddings, original_features=None, gumbel_tau=1.0, seed=42):
        x = cell_embeddings.detach()  # frozen/precomputed encoder output
        raw = x if original_features is None else original_features.detach()
        if raw.shape[0] != x.shape[0]:
            raise ValueError("scGPT embeddings and original features must have the same cell rows")
        p, usage, refinement = self.refiner(x, self.assignment_gate, seed)
        assignment, score = compute_final_st_hard_assignment(x, p, usage, self.assignment_gate, self.cfg, gumbel_tau, self.training)
        counts = assignment.sum(0); active = counts > 0
        metacells = assignment.T @ raw / counts[:, None].clamp_min(self.cfg.eps)
        logits, patient, attention, size_weight, attention_score = self.predictor(metacells, counts)
        return {"logits": logits, "hard_assignment": assignment, "hard_metacells": metacells,
            "active_metacell_mask": active, "prototypes": p, "usage": usage, "score": score,
            "pooling_attention": attention, "pooling_size_weight": size_weight,
            "pooling_attention_score": attention_score, "refinement": refinement}
