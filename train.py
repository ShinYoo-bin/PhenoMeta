#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, sys, tempfile, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phenometa_matplotlib_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "phenometa_numba_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch

from phenometa.config import ModelConfig
from phenometa.data import discover_folds, discover_repeats, load_patient_bags, split_train_validation
from phenometa.embedders import available_embedders, build_embedder
from phenometa.graph_autoencoder import GraphAEConfig, embed_split_graph_ae
from phenometa.metrics import METRICS
from phenometa.model import PhenoMetaModel
from phenometa.scgpt_embedding import DEFAULT_SCGPT_CHECKPOINT, DEFAULT_SCGPT_SOURCE
from phenometa.training import evaluate, export_split, train, train_predictor_only
from phenometa.utils import (RunDiagnostics, ResourceLogger, RuntimeSession,
                           atomic_upsert_csv, ensure_dir, logger, run_id,
                           save_json, set_seed, summarize_resource)

FOLD_COLUMNS = ["run_ts", "dataset", "foundation_model", "repeat", "fold", "seed", "n_train_patients", "n_val_patients",
 "n_test_patients", "joint_train_time_sec", "test_time_sec", "train_export_time_sec", "test_export_time_sec",
 "joint_best_epoch", "joint_best_val_auroc", "joint_best_val_loss", "joint_best_bias_weight",
 *METRICS, "initial_prototypes_mean", "final_prototypes_mean",
 "compression_ratio_mean", "refinement_rounds_mean", "removed_prototypes_total", "merged_prototypes_total",
 "empty_prototypes_mean", "predictor_only_freeze_time_sec", "predictor_only_train_time_sec",
 "predictor_only_best_epoch", "predictor_only_best_val_auroc", "predictor_only_best_val_loss",
 "predictor_only_init"]


def ids(value): return None if not value else [int(x) for x in value.split(",")]


def foundation_config(value):
    if not value:
        return {}
    path = Path(value)
    try:
        result = json.loads(path.read_text()) if path.is_file() else json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"--foundation-config must be JSON or a JSON file: {exc}") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("--foundation-config must contain a JSON object")
    return result


def cache_name(spec):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec).strip("_.") or "foundation"


def completed_results(path, dataset, foundation_model):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame(columns=FOLD_COLUMNS)
    frame = pd.read_csv(path)
    required = {"repeat", "fold", *METRICS}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=FOLD_COLUMNS)
    valid = frame[list(METRICS)].notna().all(axis=1)
    if "dataset" in frame:
        valid &= frame["dataset"].astype(str).eq(str(dataset))
    if "foundation_model" in frame:
        valid &= frame["foundation_model"].astype(str).eq(str(foundation_model))
    frame = frame.loc[valid].copy()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="coerce")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
    frame = frame.dropna(subset=["repeat", "fold"])
    frame[["repeat", "fold"]] = frame[["repeat", "fold"]].astype(int)
    return frame.drop_duplicates(["repeat", "fold"], keep="last")


def run_fold(args, repeat, fold, timestamp, resource, log, diagnostics):
    with diagnostics.stage("fold_setup", repeat=repeat, fold=fold):
        seed = args.seed + repeat * 100 + fold; set_seed(seed)
        device = torch.device(
            args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        args.resolved_device = str(device)
        root = Path(args.input_dir) / f"repeat_{repeat}"
        original_train = root / f"fold_{fold}_train.h5ad"
        original_val = root / f"fold_{fold}_val.h5ad"
        original_test = root / f"fold_{fold}_test.h5ad"
        has_explicit_val = original_val.is_file()
        embedder = build_embedder(args.foundation_model, args, args.foundation_config)
        backend_name = cache_name(args.foundation_model)
        foundation_root = (Path(args.foundation_cache_root) / (args.dataset or Path(args.input_dir).name)
                           if args.foundation_cache_root else Path(args.output_dir))
        cache = foundation_root / "foundation_embeddings" / backend_name / f"repeat_{repeat}" / f"fold_{fold}"
    # Step 1 starts here: foundation embedding followed by optional graph AE.
    # Reset per fold so every recorded peak belongs to this step only.
    resource.reset_gpu_peak()
    stage_name = f"foundation_{backend_name}_train"
    with diagnostics.stage(stage_name, repeat=repeat, fold=fold, split="train",
                           input_path=str(original_train), cache_path=str(cache / "train.h5ad")):
        resource.set_status(stage_name, -1, repeat, fold)
        train_result = embedder.prepare_split(original_train, cache / "train.h5ad",
            repeat=repeat, fold=fold, split="train", log=log)
        resource.record()
    stage_name = f"foundation_{backend_name}_test"
    with diagnostics.stage(stage_name, repeat=repeat, fold=fold, split="test",
                           input_path=str(original_test), cache_path=str(cache / "test.h5ad")):
        resource.set_status(stage_name, -1, repeat, fold)
        test_result = embedder.prepare_split(original_test, cache / "test.h5ad",
            repeat=repeat, fold=fold, split="test", log=log)
        resource.record()
    val_result = None
    if has_explicit_val:
        stage_name = f"foundation_{backend_name}_val"
        with diagnostics.stage(stage_name, repeat=repeat, fold=fold, split="validation",
                               input_path=str(original_val), cache_path=str(cache / "val.h5ad")):
            resource.set_status(stage_name, -1, repeat, fold)
            val_result = embedder.prepare_split(original_val, cache / "val.h5ad",
                repeat=repeat, fold=fold, split="validation", log=log)
            resource.record()
    with diagnostics.stage("foundation_validate", repeat=repeat, fold=fold):
        train_path, test_path = train_result.path, test_result.path
        val_path = val_result.path if val_result is not None else None
        effective_embedding_key = train_result.key
        if test_result.key != effective_embedding_key:
            raise ValueError(
                "foundation embedding keys differ: "
                f"train={effective_embedding_key} test={test_result.key}"
            )
        if val_result is not None and val_result.key != effective_embedding_key:
            raise ValueError(
                "foundation embedding keys differ: "
                f"train={effective_embedding_key} validation={val_result.key}"
            )
    if args.use_graph_ae:
        ae_cfg = GraphAEConfig(latent_dim=args.ae_latent_dim, knn_k=args.ae_knn_k,
            epochs=args.ae_epochs, lr=args.ae_lr, weight_decay=args.ae_weight_decay,
            lambda_rec=args.ae_lambda_rec, lambda_lap=args.ae_lambda_lap, lambda_con=args.ae_lambda_con,
            temperature=args.ae_temperature, negatives_per_anchor=args.ae_negatives_per_anchor,
            max_anchors=args.ae_max_anchors, early_stopping_min_delta=args.ae_early_stopping_min_delta,
            early_stopping_patience=args.ae_early_stopping_patience)
        graph_root = (Path(args.graph_ae_cache_root) / (args.dataset or Path(args.input_dir).name)
                      if args.graph_ae_cache_root else Path(args.output_dir))
        ae_cache = graph_root / "graph_ae" / f"repeat_{repeat}" / f"fold_{fold}"
        with diagnostics.stage("graph_ae_train", repeat=repeat, fold=fold, split="train",
                               input_path=str(train_path), cache_path=str(ae_cache / "train_latent.h5ad")):
            resource.set_status("graph_ae_train", -1, repeat, fold)
            train_path = embed_split_graph_ae(train_path, ae_cache / "train_latent.h5ad",
                patient_col=args.patient_col, embedding_key=effective_embedding_key, cfg=ae_cfg,
                device=device, seed=args.seed, graph_input_path=original_train,
                graph_input_key=args.original_feature_key, overwrite=args.overwrite_graph_ae,
                patient_cache_dir=graph_root / "shared_patient_cache" / "graph_ae", log=log,
                repeat=repeat, fold=fold, split="train")
            resource.record()
        with diagnostics.stage("graph_ae_test", repeat=repeat, fold=fold, split="test",
                               input_path=str(test_path), cache_path=str(ae_cache / "test_latent.h5ad")):
            resource.set_status("graph_ae_test", -1, repeat, fold)
            test_path = embed_split_graph_ae(test_path, ae_cache / "test_latent.h5ad",
                patient_col=args.patient_col, embedding_key=effective_embedding_key, cfg=ae_cfg,
                device=device, seed=args.seed, graph_input_path=original_test,
                graph_input_key=args.original_feature_key, overwrite=args.overwrite_graph_ae,
                patient_cache_dir=graph_root / "shared_patient_cache" / "graph_ae", log=log,
                repeat=repeat, fold=fold, split="test")
            resource.record()
        if has_explicit_val:
            with diagnostics.stage("graph_ae_val", repeat=repeat, fold=fold, split="validation",
                                   input_path=str(val_path), cache_path=str(ae_cache / "val_latent.h5ad")):
                resource.set_status("graph_ae_val", -1, repeat, fold)
                val_path = embed_split_graph_ae(val_path, ae_cache / "val_latent.h5ad",
                    patient_col=args.patient_col, embedding_key=effective_embedding_key, cfg=ae_cfg,
                    device=device, seed=args.seed, graph_input_path=original_val,
                    graph_input_key=args.original_feature_key, overwrite=args.overwrite_graph_ae,
                    patient_cache_dir=graph_root / "shared_patient_cache" / "graph_ae", log=log,
                    repeat=repeat, fold=fold, split="validation")
                resource.record()
        effective_embedding_key = "X_latent"
    # Persist the final Step 1 peak before resetting it for Step 2.
    resource.synchronize_cuda()
    resource.record()
    resource.reset_gpu_peak()
    with diagnostics.stage("load_patient_bags", repeat=repeat, fold=fold,
                           train_path=str(train_path), test_path=str(test_path)):
        resource.set_status("load_patient_bags", -1, repeat, fold)
        all_train, label_map = load_patient_bags(train_path, args.patient_col, args.label_col,
            effective_embedding_key, source_split="train", original_feature_path=original_train,
            original_feature_key=args.original_feature_key)
        test_bags, _ = load_patient_bags(test_path, args.patient_col, args.label_col,
            effective_embedding_key, label_map, "test", original_feature_path=original_test,
            original_feature_key=args.original_feature_key)
        if has_explicit_val:
            val_bags, _ = load_patient_bags(val_path, args.patient_col, args.label_col,
                effective_embedding_key, label_map, "validation", original_feature_path=original_val,
                original_feature_key=args.original_feature_key)
            train_bags = all_train
            validation_source = "explicit fold_val.h5ad"
        else:
            train_bags, val_bags = split_train_validation(all_train, args.val_fraction, seed)
            validation_source = f"inner split from train ({args.val_fraction:.3f})"
        log.info("[bags] repeat=%d fold=%d train=%d validation=%d test=%d", repeat, fold,
                 len(train_bags), len(val_bags), len(test_bags))
        log.info("[validation source] repeat=%d fold=%d source=%s", repeat, fold, validation_source)
        cfg = ModelConfig(latent_dim=train_bags[0].embeddings.shape[1], n_classes=len(label_map),
            metacell_feature_dim=train_bags[0].original_features.shape[1],
            init_scale=args.init_scale, min_init_prototypes=args.min_init_prototypes, max_init_prototypes=args.max_init_prototypes,
            top_k_ratio=args.top_k_ratio, min_top_k=args.min_top_k, max_top_k=args.max_top_k,
            refinement_tau=args.refinement_tau, usage_alpha=args.usage_alpha, convergence_tol=args.convergence_tol,
            max_refinement_iter=args.max_refinement_iter, dead_eps=args.dead_eps, merge_threshold=args.merge_threshold,
            merge_usage_ratio_threshold=args.merge_usage_ratio_threshold, gate_hidden_dim=args.gate_hidden_dim,
            bias_weight=args.bias_weight, use_assignment_gate=args.use_assignment_gate,
            use_bias_in_refinement=args.use_bias_in_refinement, use_bias_in_final_assignment=args.use_bias_in_final_assignment,
            use_st_gumbel_final_assignment=args.use_st_gumbel_final_assignment,
            use_interaction_pair_features=args.use_interaction_pair_features, pooling=args.pooling,
            pooling_hidden_dim=args.pooling_hidden_dim, classifier_hidden_dim=args.classifier_hidden_dim,
            predictor_encoder_dropout=args.classifier_dropout, classifier_dropout=args.classifier_dropout)
        model = PhenoMetaModel(cfg).to(device)
    checkpoint = Path(args.output_dir) / "checkpoints" / f"repeat_{repeat}_fold_{fold}.pt"
    history = Path(args.output_dir) / "training_history" / f"repeat_{repeat}_fold_{fold}.csv"; ensure_dir(history.parent)
    with diagnostics.stage("joint_training", repeat=repeat, fold=fold,
                           n_train=len(train_bags), n_validation=len(val_bags)):
        fit = train(model, train_bags, val_bags, device, checkpoint, history,
                    epochs=args.epochs, patience=args.patience,
                    learning_rate=args.lr, weight_decay=args.weight_decay,
                    clip_grad=args.clip_grad_norm, tau_start=args.gumbel_tau_start, tau_min=args.gumbel_tau_min,
                    tau_decay=args.gumbel_tau_decay, seed=seed, resource_logger=resource,
                    repeat=repeat, fold=fold, log=log, gate_warmup_epochs=args.gate_warmup_epochs,
                    gate_ramp_epochs=args.gate_ramp_epochs)
        save_json({
            "stage": "joint",
            "bias_weight": fit["joint_best_bias_weight"],
            "model_config": cfg.to_dict(),
            "label_map": label_map,
        }, checkpoint.with_suffix(".json"))
    predictor_checkpoint = (
        Path(args.output_dir) / "predictor_only_checkpoints"
        / f"repeat_{repeat}_fold_{fold}.pt"
    )
    predictor_history = (
        Path(args.output_dir) / "predictor_only_history"
        / f"repeat_{repeat}_fold_{fold}.csv"
    )
    ensure_dir(predictor_history.parent)
    with diagnostics.stage("predictor_only", repeat=repeat, fold=fold,
                           predictor_init=args.predictor_only_init,
                           n_train=len(train_bags), n_validation=len(val_bags)):
        resource.set_status("predictor_only_materialize", -1, repeat, fold)
        post_fit = train_predictor_only(
            model,
            train_bags,
            val_bags,
            device,
            predictor_checkpoint,
            predictor_history,
            epochs=args.predictor_only_epochs,
            patience=args.predictor_only_patience,
            learning_rate=args.predictor_only_lr,
            weight_decay=args.predictor_only_weight_decay,
            clip_grad=args.predictor_only_clip_grad_norm,
            seed=seed,
            predictor_init=args.predictor_only_init,
            resource_logger=resource,
            repeat=repeat,
            fold=fold,
            log=log,
            label_map=label_map,
        )
    with diagnostics.stage("test", repeat=repeat, fold=fold, split="test",
                           n_patients=len(test_bags)):
        resource.set_status("test", -1, repeat, fold); start = time.time()
        test_metrics, _, _, _ = evaluate(
            model, test_bags, device, seed, log=log, repeat=repeat, fold=fold,
            stage="test", split="test")
        test_time = time.time() - start; resource.record()
    combined = train_bags + val_bags
    with diagnostics.stage("export_train", repeat=repeat, fold=fold, split="train",
                           n_patients=len(combined)):
        resource.set_status("export_train", -1, repeat, fold)
        train_export, train_ref = export_split(
            model, combined, args.output_dir, "train", repeat, fold, device, seed, log=log)
    with diagnostics.stage("export_test", repeat=repeat, fold=fold, split="test",
                           n_patients=len(test_bags)):
        resource.set_status("export_test", -1, repeat, fold)
        test_export, test_ref = export_split(
            model, test_bags, args.output_dir, "test", repeat, fold, device, seed, log=log)
    # Persist the final Step 2 peak even when the periodic sampler did not run
    # near the short inference/export stages.
    resource.synchronize_cuda()
    resource.record()
    ref = test_ref
    return {"run_ts": timestamp, "dataset": args.dataset or Path(args.input_dir).name,
        "foundation_model": args.foundation_model, "repeat": repeat, "fold": fold,
        "seed": seed, "n_train_patients": len(train_bags), "n_val_patients": len(val_bags),
        "n_test_patients": len(test_bags), **fit, **post_fit, "test_time_sec": test_time,
        "train_export_time_sec": train_export, "test_export_time_sec": test_export, **test_metrics,
        "initial_prototypes_mean": ref.initial_prototypes.mean(), "final_prototypes_mean": ref.final_prototypes.mean(),
        "compression_ratio_mean": ref.compression_ratio.mean(), "refinement_rounds_mean": ref.refinement_rounds.mean(),
        "removed_prototypes_total": ref.removed.sum(), "merged_prototypes_total": ref.merged.sum(),
        "empty_prototypes_mean": ref.empty_prototypes.mean(), "label_map": label_map, "model_config": cfg.to_dict()}


def main():
    p = argparse.ArgumentParser(
        description=(
            "PhenoMeta joint phenotype-guided routing followed by frozen-routing "
            "size-aware attention predictor training"
        )
    )
    p.add_argument("--input-dir", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--dataset")
    p.add_argument("--patient-col", default="patient"); p.add_argument("--label-col", default="label")
    p.add_argument("--embedding-key", default="X", help="Embedding key for the precomputed backend")
    p.add_argument("--repeats"); p.add_argument("--folds")
    p.add_argument("--original-feature-key", default="X")
    p.add_argument("--foundation-model", default="scgpt",
                   help=f"Registered backend ({', '.join(available_embedders())}) or python.module:ClassName")
    p.add_argument("--foundation-config", type=foundation_config, default={},
                   help="Backend-specific JSON object or path to a JSON file")
    p.add_argument("--foundation-cache-root",
                   help="Optional common root containing <dataset>/foundation_embeddings; outputs stay in --output-dir")
    p.add_argument("--graph-ae-cache-root",
                   help="Optional common root containing <dataset>/{graph_ae,shared_patient_cache/graph_ae}")
    p.add_argument("--embedding-input-dir",
                   help="For precomputed: directory containing repeat_*/fold_*_{train,test}.h5ad")
    p.add_argument("--scgpt-source-dir", default=str(DEFAULT_SCGPT_SOURCE))
    p.add_argument("--scgpt-checkpoint-dir", default=str(DEFAULT_SCGPT_CHECKPOINT))
    p.add_argument("--scgpt-gene-col", default="auto")
    p.add_argument("--scgpt-batch-size", type=int, default=2000)
    p.add_argument("--scgpt-max-length", type=int, default=1200)
    p.add_argument("--scgpt-use-fast-transformer", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--uce-source-dir", default=os.environ.get("UCE_SOURCE_DIR", str(PROJECT_ROOT / "external" / "UCE")))
    p.add_argument("--uce-python", default=os.environ.get("UCE_PYTHON", sys.executable))
    p.add_argument("--uce-species", default="human")
    p.add_argument("--uce-gene-col", default="auto",
                   help="Gene-symbol column for UCE; auto selects the candidate with greatest protein-embedding overlap")
    p.add_argument("--uce-batch-size", type=int, default=100)
    p.add_argument("--uce-model-loc", help="Defaults to UCE's automatically downloaded 4-layer checkpoint")
    p.add_argument("--uce-nlayers", type=int, default=4)
    p.add_argument("--uce-sample-size", type=int, default=1024)
    p.add_argument("--uce-filter", action=argparse.BooleanOptionalAction, default=False,
                   help="UCE gene/cell filtering; disabled by default to preserve every split cell")
    p.add_argument("--overwrite-foundation-cache", action="store_true")
    p.add_argument("--use-graph-ae", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ae-latent-dim", type=int, default=32)
    p.add_argument("--ae-knn-k", type=int, default=5)
    p.add_argument("--ae-epochs", type=int, default=300)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-weight-decay", type=float, default=1e-2)
    p.add_argument("--ae-lambda-rec", type=float, default=.6)
    p.add_argument("--ae-lambda-lap", type=float, default=.2)
    p.add_argument("--ae-lambda-con", type=float, default=.2)
    p.add_argument("--ae-temperature", type=float, default=.2)
    p.add_argument("--ae-negatives-per-anchor", type=int, default=32)
    p.add_argument("--ae-max-anchors", type=int, default=1024)
    p.add_argument("--ae-early-stopping-min-delta", type=float, default=1e-5)
    p.add_argument("--ae-early-stopping-patience", type=int, default=10)
    p.add_argument("--overwrite-graph-ae", action="store_true")
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--val-fraction", type=float, default=.25); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad-norm", type=float, default=1.)
    p.add_argument("--gumbel-tau-start", type=float, default=1.); p.add_argument("--gumbel-tau-min", type=float, default=.1)
    p.add_argument("--gumbel-tau-decay", type=float, default=.95); p.add_argument("--init-scale", type=float, default=2.)
    p.add_argument("--min-init-prototypes", type=int, default=16); p.add_argument("--max-init-prototypes", type=int, default=2048)
    p.add_argument("--top-k-ratio", type=float, default=.05); p.add_argument("--min-top-k", type=int, default=5)
    p.add_argument("--max-top-k", type=int, default=50); p.add_argument("--refinement-tau", type=float, default=.05)
    p.add_argument("--usage-alpha", type=float, default=.001); p.add_argument("--convergence-tol", type=float, default=1e-4)
    p.add_argument("--max-refinement-iter", type=int, default=100); p.add_argument("--dead-eps", type=float, default=1e-4)
    p.add_argument("--merge-threshold", type=float, default=.99); p.add_argument("--merge-usage-ratio-threshold", type=float, default=3.)
    p.add_argument("--gate-hidden-dim", type=int, default=128); p.add_argument("--bias-weight", type=float, default=1.0)
    p.add_argument("--gate-warmup-epochs", type=int, default=10)
    p.add_argument("--gate-ramp-epochs", type=int, default=10)
    p.add_argument("--predictor-only-epochs", type=int, default=50,
                   help="Epochs for the frozen-routing predictor stage after joint training")
    p.add_argument("--predictor-only-patience", type=int, default=10)
    p.add_argument("--predictor-only-lr", type=float, default=1e-4)
    p.add_argument("--predictor-only-weight-decay", type=float, default=1e-4)
    p.add_argument("--predictor-only-clip-grad-norm", type=float, default=1.0)
    p.add_argument("--predictor-only-init", choices=["reset", "joint"], default="reset",
                   help=("reset reproduces the separate fixed-metacell predictor experiment; "
                         "joint fine-tunes the predictor learned during joint training"))
    p.add_argument("--pooling", choices=["size_aware_attention", "size_weighted", "mean"],
                   default="size_aware_attention")
    p.add_argument("--pooling-hidden-dim", type=int, default=128)
    p.add_argument("--classifier-hidden-dim", type=int, default=128); p.add_argument("--classifier-dropout", type=float, default=.1)
    for name, default in [("use-assignment-gate", True), ("use-bias-in-refinement", True),
                          ("use-bias-in-final-assignment", True), ("use-st-gumbel-final-assignment", True),
                          ("use-interaction-pair-features", False)]:
        p.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default)
    p.add_argument("--resource-interval-sec", type=float, default=30.)
    p.add_argument("--rerun-completed", action="store_true",
                   help="Run folds even when a complete row already exists (default: resume/skip)")
    args = p.parse_args()
    if args.gate_warmup_epochs < 0 or args.gate_ramp_epochs < 0:
        p.error("gate warmup/ramp epochs must be non-negative")
    if args.epochs <= args.gate_warmup_epochs:
        p.error("--epochs must be greater than --gate-warmup-epochs")
    if args.predictor_only_epochs <= 0 or args.predictor_only_patience <= 0:
        p.error("predictor-only epochs and patience must be positive")
    ensure_dir(args.output_dir); timestamp = run_id(); log = logger(Path(args.output_dir)/"stdout.out")
    dataset_name = args.dataset or Path(args.input_dir).name
    diagnostics = RunDiagnostics(args.output_dir, log, timestamp, dataset_name)
    resource = None
    session = RuntimeSession(args.output_dir, timestamp, dataset_name)
    rows = []
    try:
        with diagnostics.stage("initialization", input_dir=str(args.input_dir), output_dir=str(args.output_dir)):
            save_json(vars(args), Path(args.output_dir)/"config_used.json")
            resource = ResourceLogger(
                Path(args.output_dir)/"resource.csv", args.resource_interval_sec, args.device,
                heartbeat=session.heartbeat)
            resource.start()
        with diagnostics.stage("discover_splits", input_dir=str(args.input_dir)):
            repeats = ids(args.repeats) or discover_repeats(args.input_dir)
        for repeat in repeats:
            with diagnostics.stage(
                "discover_folds", repeat=repeat, fold=-1,
                input_dir=str(args.input_dir),
            ):
                folds = ids(args.folds) or discover_folds(args.input_dir, repeat)
            repeat_rows = []
            for fold in folds:
                existing = completed_results(
                    Path(args.output_dir) / "results_folds.csv",
                    dataset_name, args.foundation_model,
                )
                done = ((existing["repeat"] == repeat) & (existing["fold"] == fold)).any()
                if done and not args.rerun_completed:
                    log.info("[resume:skip] repeat=%d fold=%d already complete", repeat, fold)
                    continue
                log.info("Starting repeat=%d fold=%d", repeat, fold)
                row = run_fold(args, repeat, fold, timestamp, resource, log, diagnostics)
                with diagnostics.stage(
                    "persist_fold_result", repeat=repeat, fold=fold,
                    output_path=str(Path(args.output_dir) / "results_folds.csv"),
                ):
                    atomic_upsert_csv(
                        Path(args.output_dir) / "results_folds.csv",
                        row,
                        FOLD_COLUMNS,
                        ("dataset", "foundation_model", "repeat", "fold"),
                    )
                    rows.append(row)
                    repeat_rows.append(row)
            with diagnostics.stage("repeat_summary", repeat=repeat, fold=-1):
                all_results = completed_results(Path(args.output_dir)/"results_folds.csv", dataset_name, args.foundation_model)
                frame = all_results[all_results["repeat"] == repeat]
                if frame.empty:
                    continue
                summary = {"run_ts": timestamp, "dataset": dataset_name,
                    "foundation_model": args.foundation_model, "repeat": repeat}
                for metric in METRICS:
                    summary[f"{metric}_mean"], summary[f"{metric}_std"] = frame[metric].mean(), frame[metric].std()
                columns = ["run_ts", "dataset", "foundation_model", "repeat",
                           *[x for m in METRICS for x in (f"{m}_mean", f"{m}_std")]]
                atomic_upsert_csv(Path(args.output_dir)/"results_repeats.csv", summary, columns,
                                  ("dataset", "foundation_model", "repeat"))
        with diagnostics.stage("final_summary"):
            session.heartbeat()
            frame = completed_results(Path(args.output_dir)/"results_folds.csv", dataset_name, args.foundation_model)
            if frame.empty:
                raise RuntimeError("No completed fold results are available for the final summary")
            lines = [f"run_ts: {timestamp}", f"dataset: {dataset_name}",
                f"foundation_model: {args.foundation_model}", f"input_dir: {args.input_dir}", "", "Test metrics over folds"]
            lines += [f"{m}: {frame[m].mean():.6f} +/- {frame[m].std():.6f}" for m in METRICS]
            lines += [
                "",
                "Execution time",
                f"joint_train_time_sec_total: {frame.joint_train_time_sec.sum():.3f}",
                f"predictor_only_freeze_time_sec_total: {frame.predictor_only_freeze_time_sec.sum():.3f}",
                f"predictor_only_train_time_sec_total: {frame.predictor_only_train_time_sec.sum():.3f}",
                f"test_time_sec_total: {frame.test_time_sec.sum():.3f}",
                f"active_wall_time_sec_all_invocations: {session.total_active_seconds():.3f}",
            ]
            train_resource = summarize_resource(Path(args.output_dir)/"resource.csv", "train")
            predictor_resource = summarize_resource(Path(args.output_dir)/"resource.csv", "predictor_only")
            test_resource = summarize_resource(Path(args.output_dir)/"resource.csv", "test")
            lines += ["", "Resource mean/max"]
            lines += [f"joint_train_{key}: {value:.6f}" for key, value in train_resource.items()]
            lines += [f"predictor_only_{key}: {value:.6f}" for key, value in predictor_resource.items()]
            lines += [f"test_{key}: {value:.6f}" for key, value in test_resource.items()]
            Path(args.output_dir, "summary.txt").write_text("\n".join(lines)+"\n")
            save_json({"run_ts": timestamp,
                       "folds": [{k: v for k,v in r.items() if k not in ("label_map","model_config")}
                                 for r in rows]}, Path(args.output_dir)/"summary.json")
        diagnostics.complete()
        session.close("complete")
    except BaseException as exc:
        session.close("failed")
        if not diagnostics.failure_recorded:
            diagnostics.fail(exc)
            log.exception("[run:failed] exception_type=%s exception=%s", type(exc).__name__, exc)
        raise
    finally:
        if resource is not None:
            resource.stop()

if __name__ == "__main__": main()
