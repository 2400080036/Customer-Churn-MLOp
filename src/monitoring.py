"""Prediction logging, data-drift detection (PSI) and performance checks."""
import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from src.config import BINARY, CATEGORICAL, FEATURES, NUMERIC, PREDICTION_LOG, TARGET

_lock = threading.Lock()
EPS = 1e-4


def log_prediction(record: dict, probability: float, model_version: str,
                   path: Path = PREDICTION_LOG) -> None:
    """Append one prediction to logs/predictions.csv (what the model actually sees in use)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model_version": model_version, **{f: record[f] for f in FEATURES},
           "probability": round(float(probability), 4)}
    with _lock:
        new_file = not path.exists()
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def psi(expected, actual) -> float:
    """Population Stability Index: <0.10 stable, 0.10-0.25 small shift, >0.25 large shift."""
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(value: float) -> str:
    return "OK" if value < 0.10 else "WARN" if value < 0.25 else "ALERT"


def drift_report(baseline: dict, recent: pd.DataFrame, min_rows: int = 100) -> dict:
    """Compare recent inputs with the training data."""
    if len(recent) < min_rows:
        return {"status": "NOT_ENOUGH_DATA", "rows": len(recent), "min_rows": min_rows, "features": {}}
    features = {}
    for col, base in baseline["numeric"].items():
        bins = np.digitize(recent[col].dropna(), base["edges"])
        share = np.bincount(bins, minlength=len(base["edges"]) + 1) / max(len(bins), 1)
        value = psi(base["expected"], share)
        features[col] = {"psi": round(value, 4), "status": _status(value)}
    for col, base in baseline["categorical"].items():
        cats = list(base)
        share = recent[col].value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        value = psi([base[c] for c in cats], share)
        features[col] = {"psi": round(value, 4), "status": _status(value)}
    order = {"OK": 0, "WARN": 1, "ALERT": 2}
    worst = max((f["status"] for f in features.values()), key=order.get)
    return {"status": worst, "rows": len(recent), "features": features}


def performance_report(model, metadata: dict, labeled: pd.DataFrame, max_auc_drop: float = 0.05) -> dict:
    """Check the live model on newly labelled customers (needs the 'churn' column)."""
    y = (labeled[TARGET] == "Yes").astype(int)
    proba = model.predict_proba(labeled[FEATURES])[:, 1]
    auc = float(roc_auc_score(y, proba))
    recall = float(recall_score(y, proba >= metadata["threshold"]))
    baseline_auc = metadata["test_metrics"]["roc_auc"]
    status = "ALERT" if auc < baseline_auc - max_auc_drop else "OK"
    return {"status": status, "rows": len(labeled), "roc_auc": round(auc, 4),
            "recall": round(recall, 4), "baseline_roc_auc": round(baseline_auc, 4)}
