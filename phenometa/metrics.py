import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

METRICS = ["auroc", "auprc", "accuracy", "precision", "recall", "f1"]


def compute_metrics(y_true, probabilities, n_classes):
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    prediction = probabilities.argmax(1)
    out = {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, average="weighted", zero_division=0),
        "recall": recall_score(y_true, prediction, average="weighted", zero_division=0),
        "f1": f1_score(y_true, prediction, average="weighted", zero_division=0),
    }
    try:
        if n_classes == 2:
            out["auroc"] = roc_auc_score(y_true, probabilities[:, 1])
            out["auprc"] = average_precision_score(y_true, probabilities[:, 1])
        else:
            out["auroc"] = roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
            out["auprc"] = average_precision_score(np.eye(n_classes)[y_true], probabilities, average="weighted")
    except ValueError:
        out["auroc"] = out["auprc"] = np.nan
    return out
