from __future__ import annotations

import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .metrics import METRICS, compute_metrics
from .utils import atomic_write_frame, ensure_dir


@dataclass(frozen=True)
class FrozenMetacellBag:
    """Deterministic PP-aware metacells materialized from the best joint gate."""

    patient_id: str
    label: int
    metacells: torch.Tensor
    counts: torch.Tensor


def _original_to_device(value, device):
    """Materialize sparse raw expression for one patient, not a whole split."""
    if sp.issparse(value):
        value = torch.from_numpy(
            np.asarray(value.toarray(), dtype=np.float32, order="C"))
    elif not torch.is_tensor(value):
        value = torch.from_numpy(np.asarray(value, dtype=np.float32, order="C"))
    return value.to(device)


def gumbel_tau(epoch, start, minimum, decay):
    return max(minimum, start * decay ** epoch)


def gate_schedule(epoch, warmup_epochs, ramp_epochs, target_bias_weight):
    """Return (trainable, effective bias) for a zero-based epoch."""
    if epoch < warmup_epochs:
        return False, 0.0
    if ramp_epochs <= 0:
        return True, float(target_bias_weight)
    step = epoch - warmup_epochs + 1
    fraction = min(1.0, step / ramp_epochs)
    return True, float(target_bias_weight) * fraction


def set_gate_trainable(model, trainable):
    for parameter in model.assignment_gate.parameters():
        parameter.requires_grad_(trainable)


def grad_norm(module):
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None: total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5


def patient_seed(base_seed, patient_id):
    return (int(base_seed) + zlib.crc32(str(patient_id).encode("utf-8"))) % (2**31 - 1)


def _bag_size(bag):
    if hasattr(bag, "embeddings"):
        return int(bag.embeddings.shape[0]), None
    counts = getattr(bag, "counts", None)
    if counts is None:
        return -1, None
    return int(counts.sum()), int(counts.shape[0])


@contextmanager
def _patient_operation(log, stage, bag, *, repeat=-1, fold=-1, epoch=-1, split=None):
    """Log enough context to identify the active patient even after an abrupt failure."""
    patient_id = str(bag.patient_id)
    split = str(split or getattr(bag, "source_split", "unknown"))
    n_cells, n_metacells = _bag_size(bag)
    if log:
        log.info(
            "[patient:start] stage=%s repeat=%s fold=%s split=%s epoch=%s "
            "patient_id=%s n_cells=%s n_metacells=%s",
            stage, repeat, fold, split, epoch, patient_id, n_cells,
            "na" if n_metacells is None else n_metacells,
        )
    try:
        yield
    except BaseException as exc:
        if log:
            log.exception(
                "[patient:failed] stage=%s repeat=%s fold=%s split=%s epoch=%s "
                "patient_id=%s n_cells=%s n_metacells=%s exception_type=%s exception=%s",
                stage, repeat, fold, split, epoch, patient_id, n_cells,
                "na" if n_metacells is None else n_metacells,
                type(exc).__name__, exc,
            )
        raise


@torch.no_grad()
def evaluate(model, bags, device, seed=42, *, log=None, repeat=-1, fold=-1,
             epoch=-1, stage="evaluation", split=None):
    model.eval(); y_true, probabilities, losses, outputs = [], [], [], []
    for bag in bags:
        with _patient_operation(log, stage, bag, repeat=repeat, fold=fold,
                                epoch=epoch, split=split):
            out = model(bag.embeddings.to(device), _original_to_device(bag.original_features, device),
                        seed=patient_seed(seed, bag.patient_id))
            losses.append(float(F.cross_entropy(
                out["logits"][None], torch.tensor([bag.label], device=device))))
            probabilities.append(torch.softmax(out["logits"], 0).cpu().numpy())
            y_true.append(bag.label); outputs.append(out)
    metrics = compute_metrics(y_true, probabilities, model.cfg.n_classes)
    return metrics, float(np.mean(losses)), outputs, np.asarray(probabilities)


@torch.no_grad()
def materialize_deterministic_metacells(model, bags, device, seed=42, *, log=None,
                                        repeat=-1, fold=-1, stage="materialize", split=None):
    """Run routing once in eval mode and keep only active metacells on CPU."""
    model.eval()
    frozen = []
    for bag in bags:
        with _patient_operation(log, stage, bag, repeat=repeat, fold=fold,
                                epoch=-1, split=split):
            out = model(
                bag.embeddings.to(device),
                _original_to_device(bag.original_features, device),
                seed=patient_seed(seed, bag.patient_id),
            )
            counts = out["hard_assignment"].sum(0)
            active = counts > 0
            frozen.append(FrozenMetacellBag(
                patient_id=str(bag.patient_id),
                label=int(bag.label),
                metacells=out["hard_metacells"][active].detach().cpu().contiguous(),
                counts=counts[active].detach().cpu().contiguous(),
            ))
    return frozen


@torch.no_grad()
def evaluate_frozen_predictor(predictor, bags, device, n_classes, *, log=None,
                              repeat=-1, fold=-1, epoch=-1,
                              stage="predictor_only_validation", split="validation"):
    predictor.eval()
    y_true, probabilities, losses = [], [], []
    for bag in bags:
        with _patient_operation(log, stage, bag, repeat=repeat, fold=fold,
                                epoch=epoch, split=split):
            logits, *_ = predictor(
                bag.metacells.to(device),
                bag.counts.to(device),
            )
            target = torch.tensor([bag.label], device=device)
            losses.append(float(F.cross_entropy(logits[None], target)))
            probabilities.append(torch.softmax(logits, 0).cpu().numpy())
            y_true.append(bag.label)
    metrics = compute_metrics(y_true, probabilities, n_classes)
    return metrics, float(np.mean(losses)), np.asarray(probabilities)


def train_predictor_only(
    model,
    train_bags,
    val_bags,
    device,
    checkpoint,
    history_path,
    epochs=50,
    patience=10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    clip_grad=1.0,
    seed=42,
    predictor_init="reset",
    resource_logger=None,
    repeat=0,
    fold=0,
    log=None,
    label_map=None,
):
    """Freeze the best joint routing, then fit only the attention PP predictor."""
    if predictor_init not in {"reset", "joint"}:
        raise ValueError("predictor_init must be 'reset' or 'joint'")
    if epochs <= 0:
        raise ValueError("predictor-only epochs must be positive")

    freeze_start = time.time()
    set_gate_trainable(model, False)
    for parameter in model.assignment_gate.parameters():
        parameter.grad = None
    frozen_train = materialize_deterministic_metacells(
        model, train_bags, device, seed, log=log, repeat=repeat, fold=fold,
        stage="predictor_only_materialize", split="train")
    frozen_val = materialize_deterministic_metacells(
        model, val_bags, device, seed, log=log, repeat=repeat, fold=fold,
        stage="predictor_only_materialize", split="validation")
    freeze_time = time.time() - freeze_start

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if predictor_init == "reset":
        model.reset_predictor()
    set_gate_trainable(model, False)
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_score, best_epoch, best_loss, best_auroc, stale, history = (
        -np.inf, -1, np.inf, np.nan, 0, []
    )
    history_columns = ["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc",
                       "routing", "predictor_init", "bias_weight"]
    atomic_write_frame(pd.DataFrame(columns=history_columns), history_path)
    start = time.time()
    for epoch in range(epochs):
        if resource_logger:
            resource_logger.set_status("predictor_only_fit", epoch, repeat, fold)
        model.predictor.train()
        order = np.random.default_rng(seed + epoch).permutation(len(frozen_train))
        losses = []
        for position in order:
            bag = frozen_train[int(position)]
            with _patient_operation(log, "predictor_only_fit", bag, repeat=repeat,
                                    fold=fold, epoch=epoch + 1, split="train"):
                optimizer.zero_grad(set_to_none=True)
                logits, *_ = model.predictor(
                    bag.metacells.to(device),
                    bag.counts.to(device),
                )
                loss = F.cross_entropy(
                    logits[None],
                    torch.tensor([bag.label], device=device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), clip_grad)
                optimizer.step()
                losses.append(float(loss.detach()))

        if resource_logger:
            resource_logger.set_status("predictor_only_validation", epoch, repeat, fold)
        val_metrics, val_loss, _ = evaluate_frozen_predictor(
            model.predictor, frozen_val, device, model.cfg.n_classes, log=log,
            repeat=repeat, fold=fold, epoch=epoch + 1,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "routing": "frozen_deterministic_argmax",
            "predictor_init": predictor_init,
            "bias_weight": float(model.cfg.bias_weight),
        }
        history.append(row)
        atomic_write_frame(pd.DataFrame(history, columns=history_columns), history_path)
        if log:
            log.info(
                "predictor_only epoch=%d train_loss=%.6f val_loss=%.6f "
                "val_auroc=%s init=%s bias=%.3f",
                epoch + 1,
                row["train_loss"],
                val_loss,
                val_metrics["auroc"],
                predictor_init,
                model.cfg.bias_weight,
            )

        score = val_metrics["auroc"] if np.isfinite(val_metrics["auroc"]) else -val_loss
        if score > best_score:
            best_score, best_epoch, best_loss, best_auroc, stale = (
                score, epoch + 1, val_loss, val_metrics["auroc"], 0
            )
            ensure_dir(Path(checkpoint).parent)
            torch.save({
                "stage": "post_joint_predictor_only",
                "epoch": best_epoch,
                "bias_weight": float(model.cfg.bias_weight),
                "predictor_init": predictor_init,
                "model_config": model.cfg.to_dict(),
                "label_map": dict(label_map or {}),
                "model_state_dict": model.state_dict(),
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break

    atomic_write_frame(pd.DataFrame(history, columns=history_columns), history_path)
    if resource_logger:
        resource_logger.record()
    if best_epoch < 0:
        raise RuntimeError("No predictor-only checkpoint was saved")

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.cfg.bias_weight = float(saved["bias_weight"])
    set_gate_trainable(model, False)
    model.eval()
    return {
        "predictor_only_freeze_time_sec": freeze_time,
        "predictor_only_train_time_sec": time.time() - start,
        "predictor_only_best_epoch": best_epoch,
        "predictor_only_best_val_auroc": best_auroc,
        "predictor_only_best_val_loss": best_loss,
        "predictor_only_init": predictor_init,
    }


def train(model, train_bags, val_bags, device, checkpoint, history_path, epochs=100, patience=15,
          learning_rate=1e-4, weight_decay=1e-4, clip_grad=1.0, tau_start=1.0, tau_min=0.1,
          tau_decay=0.95, seed=42, resource_logger=None, repeat=0, fold=0, log=None,
          gate_warmup_epochs=10, gate_ramp_epochs=10):
    optimizer = torch.optim.AdamW(
        list(model.assignment_gate.parameters()) + list(model.predictor.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_score, best_epoch, best_loss, best_auroc, best_bias_weight, stale, history = (
        -np.inf, -1, np.inf, np.nan, 0.0, 0, []
    )
    target_bias_weight = float(model.cfg.bias_weight)
    history_columns = ["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc",
                       "gumbel_tau", "gate_trainable", "effective_bias_weight",
                       "gate_grad_norm"]
    atomic_write_frame(pd.DataFrame(columns=history_columns), history_path)
    start = time.time()
    for epoch in range(epochs):
        gate_trainable, effective_bias_weight = gate_schedule(
            epoch, gate_warmup_epochs, gate_ramp_epochs, target_bias_weight)
        set_gate_trainable(model, gate_trainable)
        model.cfg.bias_weight = effective_bias_weight
        if resource_logger: resource_logger.set_status("train", epoch, repeat, fold)
        model.train(); order = np.random.default_rng(seed + epoch).permutation(len(train_bags)); losses, gate_norms = [], []
        tau = gumbel_tau(epoch, tau_start, tau_min, tau_decay)
        for position in order:
            bag = train_bags[int(position)]
            with _patient_operation(log, "joint_train", bag, repeat=repeat, fold=fold,
                                    epoch=epoch + 1, split="train"):
                optimizer.zero_grad(set_to_none=True)
                out = model(bag.embeddings.to(device), _original_to_device(bag.original_features, device), tau,
                            patient_seed(seed, bag.patient_id))
                loss = F.cross_entropy(
                    out["logits"][None], torch.tensor([bag.label], device=device))
                loss.backward()
                if (gate_trainable and effective_bias_weight > 0 and model.cfg.use_assignment_gate
                        and model.cfg.use_bias_in_final_assignment and model.cfg.use_st_gumbel_final_assignment):
                    if model.assignment_gate.output_layer.weight.grad is None:
                        raise RuntimeError("PP loss did not reach the assignment gate")
                gate_norms.append(grad_norm(model.assignment_gate))
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step(); losses.append(float(loss.detach()))
        if resource_logger: resource_logger.set_status("validation", epoch, repeat, fold)
        val_metrics, val_loss, _, _ = evaluate(
            model, val_bags, device, seed, log=log, repeat=repeat, fold=fold,
            epoch=epoch + 1, stage="joint_validation", split="validation")
        row = {"epoch": epoch + 1, "train_loss": np.mean(losses), "val_loss": val_loss,
               "val_auroc": val_metrics["auroc"], "val_auprc": val_metrics["auprc"],
               "gumbel_tau": tau, "gate_trainable": gate_trainable,
               "effective_bias_weight": effective_bias_weight, "gate_grad_norm": np.mean(gate_norms)}
        history.append(row)
        atomic_write_frame(pd.DataFrame(history, columns=history_columns), history_path)
        if log: log.info("epoch=%d train_loss=%.6f val_loss=%.6f val_auroc=%s tau=%.4f gate=%s bias=%.3f gate_grad=%.6f",
                         epoch + 1, row["train_loss"], val_loss, val_metrics["auroc"], tau,
                         "on" if gate_trainable else "off", effective_bias_weight, row["gate_grad_norm"])
        score = val_metrics["auroc"] if np.isfinite(val_metrics["auroc"]) else -val_loss
        eligible = epoch >= gate_warmup_epochs
        if eligible and score > best_score:
            best_score, best_epoch, best_loss, best_auroc, best_bias_weight, stale = (
                score, epoch + 1, val_loss, val_metrics["auroc"],
                effective_bias_weight, 0)
            ensure_dir(Path(checkpoint).parent); torch.save(model.state_dict(), checkpoint)
        elif eligible:
            stale += 1
            if stale >= patience: break
    atomic_write_frame(pd.DataFrame(history, columns=history_columns), history_path)
    if resource_logger: resource_logger.record()
    if best_epoch < 0:
        raise RuntimeError("No gate-enabled epoch was available for checkpoint selection; increase --epochs")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.cfg.bias_weight = best_bias_weight
    set_gate_trainable(model, True)
    return {
        "joint_train_time_sec": time.time() - start,
        "joint_best_epoch": best_epoch,
        "joint_best_val_auroc": best_auroc,
        "joint_best_val_loss": best_loss,
        "joint_best_bias_weight": best_bias_weight,
    }


@torch.no_grad()
def export_split(model, bags, output_root, name, repeat, fold, device, seed=42, log=None):
    model.eval(); output_root = Path(output_root)
    assignment_dir = output_root / "assignments" / f"repeat_{repeat}_fold_{fold}_{name}_assignments"
    ensure_dir(assignment_dir); rows, metacell_x, metacell_obs, refinement_rows, iteration_rows = [], [], [], [], []
    start = time.time()
    for bag in bags:
        with _patient_operation(log, f"export_{name}", bag, repeat=repeat, fold=fold,
                                epoch=-1, split=name):
            out = model(bag.embeddings.to(device), _original_to_device(bag.original_features, device),
                        seed=patient_seed(seed, bag.patient_id))
            hard = out["hard_assignment"].argmax(1).cpu().numpy()
            active = out["active_metacell_mask"].cpu().numpy()
            active_metacell_ids = np.flatnonzero(active)
            pooling_attention = out["pooling_attention"].cpu().numpy()
            pooling_size_weight = out["pooling_size_weight"].cpu().numpy()
            pooling_attention_score = out["pooling_attention_score"].cpu().numpy()
            score = out["score"]
            np.savez_compressed(assignment_dir / f"{bag.patient_id}_assignment.npz", patient_id=bag.patient_id,
                obs_names=bag.obs_names, hard_assignment=hard, topk_indices=score["indices"].cpu().numpy(),
                base_scores=score["base"].cpu().numpy(), bias_scores=score["bias"].cpu().numpy(),
                final_scores=score["logits"].cpu().numpy(), usage=out["usage"].cpu().numpy(), active_mask=active,
                active_metacell_ids=active_metacell_ids, pooling_attention=pooling_attention,
                pooling_size_weight=pooling_size_weight, pooling_attention_score=pooling_attention_score,
                source_split=np.asarray(bag.source_split))
            probability = torch.softmax(out["logits"], 0).cpu().numpy()
            row = {"repeat": repeat, "fold": fold, "split": name, "source_split": bag.source_split,
                   "patient": bag.patient_id, "true_id": bag.label, "true_label": bag.raw_label,
                   "pred_id": int(probability.argmax()), "pred_prob": float(probability.max())}
            row.update({f"prob_{i}": float(value) for i, value in enumerate(probability)}); rows.append(row)
            for active_index, local_id in enumerate(active_metacell_ids):
                metacell_x.append(out["hard_metacells"][local_id].cpu().numpy())
                metacell_obs.append({"patient": bag.patient_id, "metacell_local_id": int(local_id), "label": bag.raw_label,
                                     "source_split": bag.source_split, "n_cells": int((hard == local_id).sum()),
                                     "size_weight": float(pooling_size_weight[active_index]),
                                     "attention_score": float(pooling_attention_score[active_index]),
                                     "attention_weight": float(pooling_attention[active_index])})
            ref = out["refinement"]
            candidate_probability = torch.softmax(score["logits"] / model.cfg.refinement_tau, 1)
            entropy = (-(candidate_probability * candidate_probability.clamp_min(model.cfg.eps).log()).sum(1)).mean()
            refinement_rows.append({"repeat": repeat, "fold": fold, "split": name, "patient": bag.patient_id,
                **{key: ref[key] for key in ("initial_prototypes", "final_prototypes", "refinement_rounds", "removed", "merged")},
                "active_metacells": int(active.sum()), "empty_prototypes": int((~active).sum()),
                "compression_ratio": float(active.sum() / len(hard)), "usage_mean": float(out["usage"].mean()),
                "usage_std": float(out["usage"].std(unbiased=False)), "usage_min": float(out["usage"].min()),
                "usage_max": float(out["usage"].max()), "assignment_entropy": float(entropy),
                "base_score_mean": float(score["base"].mean()), "base_score_std": float(score["base"].std(unbiased=False)),
                "bias_mean": float(score["bias"].mean()), "bias_std": float(score["bias"].std(unbiased=False)),
                "bias_min": float(score["bias"].min()), "bias_max": float(score["bias"].max()),
                "weighted_bias_mean": float((model.cfg.bias_weight * score["bias"]).mean()),
                "weighted_bias_std": float((model.cfg.bias_weight * score["bias"]).std(unbiased=False)),
                "final_score_mean": float(score["logits"].mean()), "final_score_std": float(score["logits"].std(unbiased=False)),
                "bias_weight": model.cfg.bias_weight})
            for item in ref["iterations"]:
                iteration_rows.append({"repeat": repeat, "fold": fold, "split": name,
                                       "patient": bag.patient_id, **item})
    prediction_dir = output_root / "predictions"; refinement_dir = output_root / "refinement_logs"
    ensure_dir(prediction_dir); ensure_dir(refinement_dir)
    pd.DataFrame(rows).to_csv(prediction_dir / f"repeat_{repeat}_fold_{fold}_{name}_predictions.csv", index=False)
    pd.DataFrame(refinement_rows).to_csv(refinement_dir / f"repeat_{repeat}_fold_{fold}_{name}.csv", index=False)
    pd.DataFrame(iteration_rows).to_csv(refinement_dir / f"repeat_{repeat}_fold_{fold}_{name}_iterations.csv", index=False)
    metacell_dir = output_root / "metacells"; ensure_dir(metacell_dir)
    ad.AnnData(np.asarray(metacell_x, dtype=np.float32), obs=pd.DataFrame(metacell_obs)).write_h5ad(
        metacell_dir / f"repeat_{repeat}_fold_{fold}_{name}_metacells.h5ad")
    return time.time() - start, pd.DataFrame(refinement_rows)
