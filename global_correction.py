#!/usr/bin/env python3
"""Add a validation-gated pseudobulk ridge correction to frozen PhenoMeta predictions."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phenometa.utils import RuntimeSession

METRICS = ("auroc", "auprc", "accuracy", "precision", "recall", "f1")
DEFAULT_CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)
DEFAULT_ALPHAS = (0.0, 0.05, 0.1, 0.25, 0.5)


def atomic_frame(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def read_frame(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def upsert_frame(path, rows, keys=("repeat", "fold")):
    current = read_frame(path)
    incoming = pd.DataFrame(rows)
    combined = pd.concat([current, incoming], ignore_index=True, sort=False)
    if not combined.empty and set(keys).issubset(combined.columns):
        combined = combined.drop_duplicates(list(keys), keep="last").sort_values(list(keys))
    atomic_frame(combined, path)
    return combined


def completed_jobs(output):
    output = Path(output)
    results = read_frame(output / "results_folds.csv")
    selections = read_frame(output / "selection.csv")
    required = {"repeat", "fold", *METRICS}
    if not required.issubset(results.columns) or not {"repeat", "fold"}.issubset(selections.columns):
        return set()
    selection_pairs = set(zip(selections.repeat.astype(int), selections.fold.astype(int)))
    done = set()
    for row in results.dropna(subset=list(METRICS)).itertuples(index=False):
        pair = (int(row.repeat), int(row.fold))
        prediction = output / "predictions" / f"repeat_{pair[0]}_fold_{pair[1]}_test_predictions.csv"
        if pair in selection_pairs and prediction.exists() and prediction.stat().st_size:
            done.add(pair)
    return done


def probability_columns(frame):
    columns = sorted((column for column in frame if column.startswith("prob_")),
                     key=lambda column: int(column.split("_")[-1]))
    if len(columns) < 2 or columns != [f"prob_{index}" for index in range(len(columns))]:
        raise KeyError("Predictions require contiguous prob_0 ... prob_C columns")
    return columns


def metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if probability.ndim != 2:
        raise ValueError("probability must have shape [patients, classes]")
    prediction = probability.argmax(1)
    if probability.shape[1] == 2:
        auroc = roc_auc_score(y, probability[:, 1])
        auprc = average_precision_score(y, probability[:, 1])
    else:
        one_hot = np.eye(probability.shape[1])[y]
        auroc = roc_auc_score(y, probability, multi_class="ovr", average="weighted")
        auprc = average_precision_score(one_hot, probability, average="weighted")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, average="weighted", zero_division=0),
        "recall": recall_score(y, prediction, average="weighted", zero_division=0),
        "f1": f1_score(y, prediction, average="weighted", zero_division=0),
    }


def raw_patient_means(path, patient_col):
    data = ad.read_h5ad(path)
    result = {}
    for patient, indices in data.obs.groupby(patient_col, sort=True, observed=False).indices.items():
        values = data.X[np.asarray(indices, dtype=np.int64)]
        mean = np.asarray(values.mean(0)).reshape(-1) if sparse.issparse(values) else values.mean(0)
        result[str(patient)] = np.asarray(mean, dtype=np.float64)
    return result


def metacell_patient_means(path, patient_col):
    data = ad.read_h5ad(path)
    if "n_cells" not in data.obs:
        raise KeyError(f"Cached metacells have no obs['n_cells']: {path}")
    result = {}
    for patient, indices in data.obs.groupby(patient_col, sort=True, observed=False).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        values = data.X[indices]
        if sparse.issparse(values):
            values = values.toarray()
        counts = data.obs.iloc[indices]["n_cells"].to_numpy(dtype=np.float64)
        result[str(patient)] = np.average(np.asarray(values, dtype=np.float64), axis=0, weights=counts)
    return result


def cached_patient_means(source_path, cache_path, patient_col, source_kind):
    source_path, cache_path = Path(source_path), Path(cache_path)
    stat = source_path.stat()
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=False)
        valid = (str(saved["source_path"].item()) == str(source_path.resolve()) and
                 int(saved["source_size"].item()) == stat.st_size and
                 int(saved["source_mtime_ns"].item()) == stat.st_mtime_ns and
                 str(saved["patient_col"].item()) == patient_col and
                 str(saved["source_kind"].item()) == source_kind)
        if valid:
            return {str(patient): feature for patient, feature in zip(saved["patients"], saved["features"])}, True
    means = (metacell_patient_means(source_path, patient_col) if source_kind == "metacell"
             else raw_patient_means(source_path, patient_col))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, patients=np.asarray(list(means)), features=np.stack(list(means.values())),
                        source_path=str(source_path.resolve()), source_size=stat.st_size,
                        source_mtime_ns=stat.st_mtime_ns, patient_col=patient_col, source_kind=source_kind)
    return means, False


def resolve_metacell_root(attention_output_dir):
    direct = Path(attention_output_dir) / "metacells"
    if direct.is_dir():
        return direct
    config_path = Path(attention_output_dir) / "config_used.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    candidates = [config.get("joint_output_dir"), config.get("source_output_dir")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "metacells").is_dir():
            return Path(candidate) / "metacells"
    return None


def mean_sources(input_dir, metacell_root, repeat, fold):
    prefix = f"repeat_{repeat}_fold_{fold}"
    if metacell_root is not None:
        train = metacell_root / f"{prefix}_train_metacells.h5ad"
        test = metacell_root / f"{prefix}_test_metacells.h5ad"
        if train.exists() and test.exists():
            return train, train, test, "metacell"
    fold_root = Path(input_dir) / f"repeat_{repeat}"
    train = fold_root / f"fold_{fold}_train.h5ad"
    validation = fold_root / f"fold_{fold}_val.h5ad"
    if not validation.exists():
        validation = train
    test = fold_root / f"fold_{fold}_test.h5ad"
    return train, validation, test, "raw"


def matrix(frame, means):
    missing = sorted(set(frame.patient.astype(str)).difference(means))
    if missing:
        raise KeyError(f"Patients missing from expression input: {missing[:5]}")
    return np.stack([means[str(patient)] for patient in frame.patient])


def read_predictions(root, repeat, fold):
    prefix = f"repeat_{repeat}_fold_{fold}"
    train = pd.read_csv(root / f"{prefix}_train_predictions.csv")
    validation_path = root / f"{prefix}_validation_predictions.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        train = train[~train.patient.astype(str).isin(validation.patient.astype(str))].copy()
    elif "source_split" in train:
        validation = train[train.source_split == "validation"].copy()
        train = train[train.source_split == "train"].copy()
    else:
        raise FileNotFoundError(f"No validation predictions for {prefix}")
    test = pd.read_csv(root / f"{prefix}_test_predictions.csv")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        required = {"patient", "true_id"}
        if not required.issubset(frame):
            raise KeyError(f"{name} predictions missing {sorted(required.difference(frame))}")
        if frame.patient.astype(str).duplicated().any():
            raise ValueError(f"Duplicate patients in {name} predictions")
    columns = probability_columns(train)
    if probability_columns(validation) != columns or probability_columns(test) != columns:
        raise ValueError("Train/validation/test probability columns differ")
    return train, validation, test


def ridge_model(c, seed, n_classes):
    solver = "liblinear" if n_classes == 2 else "lbfgs"
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", LogisticRegression(C=c, penalty="l2", solver=solver,
                                     class_weight="balanced", max_iter=5000,
                                     random_state=seed)),
    ])


def select_ridge(x_train, y_train, x_validation, y_validation, cs, seed):
    candidates = []
    n_classes = len(np.unique(y_train))
    for c in cs:
        model = ridge_model(c, seed, n_classes).fit(x_train, y_train)
        probability = model.predict_proba(x_validation)
        # AUROC first; log-loss breaks ties; stronger regularization breaks exact ties.
        score = (metrics(y_validation, probability)["auroc"],
                 -log_loss(y_validation, probability, labels=np.arange(n_classes)), -c)
        candidates.append((*score, model, probability))
    return max(candidates, key=lambda row: row[:3])


def combine_probabilities(attention_probability, ridge_probability, alpha):
    attention_probability = np.clip(np.asarray(attention_probability, dtype=float), 1e-7, 1.0)
    ridge_probability = np.clip(np.asarray(ridge_probability, dtype=float), 1e-7, 1.0)
    if attention_probability.shape != ridge_probability.shape:
        raise ValueError("Attention and ridge class probabilities must match")
    return softmax(np.log(attention_probability) + alpha * np.log(ridge_probability), axis=1)


def select_alpha(y, attention_probability, ridge_probability, alphas):
    candidates = []
    for alpha in alphas:
        probability = combine_probabilities(attention_probability, ridge_probability, alpha)
        # The safety rule is intentional: equal AUROC always falls back to smaller alpha.
        score = (metrics(y, probability)["auroc"], -alpha,
                 -log_loss(y, probability, labels=np.arange(probability.shape[1])))
        candidates.append((*score, alpha, probability))
    return max(candidates, key=lambda row: row[:3])


def discover_jobs(input_dir, repeat_filter, fold_filter):
    jobs = []
    for repeat_dir in sorted(Path(input_dir).glob("repeat_*")):
        repeat = int(repeat_dir.name.split("_")[-1])
        if repeat_filter and repeat not in repeat_filter:
            continue
        for path in sorted(repeat_dir.glob("fold_*_train.h5ad")):
            fold = int(path.name.split("_")[1])
            if (not fold_filter or fold in fold_filter) and (repeat_dir / f"fold_{fold}_test.h5ad").exists():
                jobs.append((repeat, fold))
    return jobs


def run(args):
    invocation_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    invocation_start = time.time()
    output = Path(args.output_dir)
    prediction_output = output / "predictions"
    prediction_output.mkdir(parents=True, exist_ok=True)
    prediction_root = Path(args.attention_output_dir) / "predictions"
    cache_root = Path(args.cache_dir) if args.cache_dir else output / "cache" / "pseudobulk"
    metacell_root = resolve_metacell_root(args.attention_output_dir)
    rows, selections = [], []
    jobs = discover_jobs(args.input_dir, args.repeats, args.folds)
    if not jobs:
        raise ValueError("No repeat/fold jobs discovered")
    done = completed_jobs(output)
    pending = jobs if args.rerun_completed else [job for job in jobs if job not in done]
    session_path = output / "run_sessions.csv"
    session = {"invocation_id": invocation_id, "started_at": datetime.datetime.now().isoformat(),
               "ended_at": "", "status": "running", "requested_jobs": len(jobs),
               "completed_before": len(done.intersection(jobs)), "executed_jobs": 0,
               "skipped_jobs": len(jobs) - len(pending), "elapsed_sec": 0.0}
    upsert_frame(session_path, [session], keys=("invocation_id",))
    for repeat, fold in pending:
        fold_start = time.time()
        train, validation, test = read_predictions(prediction_root, repeat, fold)
        probability_cols = probability_columns(train)
        train_source, validation_source, test_source, source_kind = mean_sources(
            args.input_dir, metacell_root, repeat, fold)
        cache_prefix = cache_root / f"repeat_{repeat}" / f"fold_{fold}"
        train_means, train_cache_hit = cached_patient_means(
            train_source, cache_prefix.with_name(cache_prefix.name + "_train.npz"), args.patient_col, source_kind)
        if validation_source == train_source:
            validation_means, validation_cache_hit = train_means, train_cache_hit
        else:
            validation_means, validation_cache_hit = cached_patient_means(
                validation_source, cache_prefix.with_name(cache_prefix.name + "_val.npz"),
                args.patient_col, source_kind)
        test_means, test_cache_hit = cached_patient_means(
            test_source, cache_prefix.with_name(cache_prefix.name + "_test.npz"), args.patient_col, source_kind)
        x_train, x_validation, x_test = (matrix(train, train_means), matrix(validation, validation_means),
                                         matrix(test, test_means))
        y_train, y_validation, y_test = (frame.true_id.to_numpy(int) for frame in (train, validation, test))
        seed = args.seed + repeat * 1000 + fold
        ridge_val_auc, neg_ridge_loss, neg_c, model, ridge_val_probability = select_ridge(
            x_train, y_train, x_validation, y_validation, args.cs, seed)
        ridge_validation_probability = model.predict_proba(x_validation)
        ridge_test_probability = model.predict_proba(x_test)
        combined_val_auc, _, neg_combined_loss, alpha, _ = select_alpha(
            y_validation, validation[probability_cols].to_numpy(float),
            ridge_validation_probability, args.alphas)
        attention_test_probability = test[probability_cols].to_numpy(float)
        combined_probability = combine_probabilities(
            attention_test_probability, ridge_test_probability, alpha)
        combined_metrics, ridge_metrics, attention_metrics = (
            metrics(y_test, combined_probability), metrics(y_test, ridge_test_probability),
            metrics(y_test, attention_test_probability))
        rows.append({"repeat": repeat, "fold": fold, "selected_C": -neg_c,
                     "selected_alpha": alpha, "ridge_val_auroc": ridge_val_auc,
                     "combined_val_auroc": combined_val_auc,
                     "correction_time_sec": time.time() - fold_start,
                     **combined_metrics, **{f"ridge_{k}": v for k, v in ridge_metrics.items()},
                     **{f"attention_{k}": v for k, v in attention_metrics.items()}})
        selections.append({"repeat": repeat, "fold": fold, "selected_C": -neg_c,
                           "ridge_val_auroc": ridge_val_auc, "ridge_val_log_loss": -neg_ridge_loss,
                           "selected_alpha": alpha, "combined_val_auroc": combined_val_auc,
                           "combined_val_log_loss": -neg_combined_loss, "mean_source": source_kind,
                           "train_mean_cache_hit": train_cache_hit,
                           "validation_mean_cache_hit": validation_cache_hit,
                           "test_mean_cache_hit": test_cache_hit})
        exported = test.copy()
        exported["selected_C"], exported["selected_alpha"] = -neg_c, alpha
        for class_id in range(len(probability_cols)):
            exported[f"ridge_prob_{class_id}"] = ridge_test_probability[:, class_id]
            exported[f"combined_prob_{class_id}"] = combined_probability[:, class_id]
        exported["combined_pred_id"] = combined_probability.argmax(1)
        exported.to_csv(prediction_output / f"repeat_{repeat}_fold_{fold}_test_predictions.csv", index=False)
        upsert_frame(output / "results_folds.csv", [rows[-1]])
        upsert_frame(output / "selection.csv", [selections[-1]])
        session["executed_jobs"] += 1
        session["elapsed_sec"] = time.time() - invocation_start
        upsert_frame(session_path, [session], keys=("invocation_id",))

    folds = read_frame(output / "results_folds.csv")
    summary = {}
    for prefix in ("", "ridge_", "attention_"):
        for metric_name in METRICS:
            column = prefix + metric_name
            summary[column + "_mean"] = float(folds[column].mean())
            summary[column + "_std"] = float(folds[column].std(ddof=1))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["PhenoMeta: frozen attention + validation-gated pseudobulk ridge correction", ""]
    for title, prefix in (("Combined", ""), ("Pseudobulk ridge only", "ridge_"),
                          ("Frozen attention baseline", "attention_")):
        lines.append(title)
        lines.extend(f"{m}: {summary[prefix+m+'_mean']:.6f} +/- {summary[prefix+m+'_std']:.6f}" for m in METRICS)
        lines.append("")
    (output / "summary.txt").write_text("\n".join(lines))
    config = vars(args).copy()
    config["jobs"] = jobs
    (output / "config_used.json").write_text(json.dumps(config, indent=2) + "\n")
    session.update(ended_at=datetime.datetime.now().isoformat(), status="complete",
                   elapsed_sec=time.time() - invocation_start)
    upsert_frame(session_path, [session], keys=("invocation_id",))


def parse_csv_ints(value):
    return [] if not value else [int(item) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--attention-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--cache-dir", default=None,
                        help="Persistent pseudobulk cache; defaults to OUTPUT_DIR/cache/pseudobulk")
    parser.add_argument("--repeats", type=parse_csv_ints, default=[])
    parser.add_argument("--folds", type=parse_csv_ints, default=[])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cs", type=float, nargs="+", default=list(DEFAULT_CS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    if 0.0 not in args.alphas or any(alpha < 0 or alpha > 0.5 for alpha in args.alphas):
        parser.error("--alphas must contain 0 and remain within [0, 0.5]")
    invocation_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session = RuntimeSession(args.output_dir, invocation_id, Path(args.input_dir).name)
    try:
        run(args)
    except BaseException:
        session.close("failed")
        raise
    else:
        session.close("complete")


if __name__ == "__main__":
    main()
