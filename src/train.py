"""Train, compare and save the churn model.

Steps
1. Split the data once (the test part is locked away).
2. Compare 3 models with 5-fold cross-validation on the TRAINING part only.
3. Calibrate the best model so probabilities are realistic.
4. Choose the decision threshold from cross-validated training predictions.
5. Evaluate ONCE on the untouched test part and save everything (model + metadata.json).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (BINARY, CATEGORICAL, DATA_CARD_PATH, DATA_PATH, FEATURES, MODEL_DIR,
                        NUMERIC, RANDOM_STATE)
from src.data import load_data, split


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + model in one object, so training and prediction always match."""
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])


def candidates() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=200, random_state=RANDOM_STATE),
    }


def calibrated(estimator) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(build_pipeline(estimator), method="sigmoid", cv=3)


def best_threshold(y_true, proba) -> float:
    grid = np.arange(0.10, 0.71, 0.01)
    return float(grid[int(np.argmax([f1_score(y_true, proba >= t) for t in grid]))])


def drift_baseline(X: pd.DataFrame) -> dict:
    """Summary of the training data, used later to detect data drift."""
    baseline = {"numeric": {}, "categorical": {}}
    for col in NUMERIC + BINARY:
        edges = np.unique(np.quantile(X[col].dropna(), np.linspace(0, 1, 11)))[1:-1]
        bins = np.digitize(X[col].dropna(), edges)
        share = np.bincount(bins, minlength=len(edges) + 1) / len(bins)
        baseline["numeric"][col] = {"edges": edges.tolist(), "expected": share.tolist()}
    for col in CATEGORICAL:
        baseline["categorical"][col] = X[col].value_counts(normalize=True).to_dict()
    return baseline


def train(model_dir: Path = MODEL_DIR, data_path: Path = DATA_PATH,
          use_mlflow: bool = True, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split(df)
    say(f"Rows: {len(df)} | train {len(X_train)} | test {len(X_test)} | "
        f"churn rate {y_train.mean():.1%}")

    # 1) compare models with cross-validation (training data only)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, est in candidates().items():
        scores = cross_val_score(build_pipeline(clone(est)), X_train, y_train,
                                 cv=cv, scoring="roc_auc")
        cv_results[name] = {"roc_auc_mean": float(scores.mean()), "roc_auc_std": float(scores.std())}
        say(f"  CV ROC-AUC  {name:20s} {scores.mean():.3f} +/- {scores.std():.3f}")
    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    say(f"Best model: {best_name}")

    # 2) calibrate + choose threshold from out-of-fold predictions
    best_est = candidates()[best_name]
    oof = cross_val_predict(calibrated(clone(best_est)), X_train, y_train, cv=cv,
                            method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof)
    model = calibrated(clone(best_est)).fit(X_train, y_train)

    # 3) one honest evaluation on the untouched test data
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    rng = np.random.default_rng(RANDOM_STATE)
    y_arr = y_test.to_numpy()
    boot = []
    for _ in range(300):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[idx].min() != y_arr[idx].max():
            boot.append(roc_auc_score(y_arr[idx], proba[idx]))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test_metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "baseline_accuracy_always_stay": float(1 - y_test.mean()),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "roc_auc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "pr_auc": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    test_metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in test_metrics.items()}
    say(f"Test ({len(X_test)} customers, threshold {threshold:.2f}): "
        f"ROC-AUC {test_metrics['roc_auc']:.3f}, recall {test_metrics['recall']:.2f}, "
        f"precision {test_metrics['precision']:.2f}, accuracy {test_metrics['accuracy']:.3f} "
        f"(always predicting 'stay' scores {test_metrics['baseline_accuracy_always_stay']:.3f})")

    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE)
    importance = dict(sorted(zip(FEATURES, map(float, imp.importances_mean)),
                             key=lambda kv: -kv[1]))

    reference = {c: float(X_train[c].median()) for c in NUMERIC + BINARY}
    reference.update({c: X_train[c].mode()[0] for c in CATEGORICAL})
    reference["senior_citizen"] = int(round(reference["senior_citizen"]))

    card = json.loads(DATA_CARD_PATH.read_text()) if DATA_CARD_PATH.exists() else {}
    digest = hashlib.sha1(Path(data_path).read_bytes()).hexdigest()[:8]
    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": f"{now:%Y%m%d}-{digest}",
        "trained_at": now.isoformat(timespec="seconds"),
        "model_name": best_name,
        "data_source": card.get("source", "User-provided csv"),
        "data_is_real": card.get("is_real", None),
        "rows": len(df), "train_rows": len(X_train), "test_rows": len(X_test),
        "churn_rate": float(y_train.mean()),
        "features": FEATURES,
        "threshold": threshold,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "permutation_importance": importance,
        "reference_customer": reference,
        "drift_baseline": drift_baseline(X_train),
    }
    joblib.dump(model, model_dir / "churn_model.pkl")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    say(f"Saved model {metadata['model_version']} to {model_dir}")

    if use_mlflow:
        log_to_mlflow(metadata, model_dir)
    return metadata


def log_to_mlflow(meta: dict, model_dir: Path) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed - skipping experiment tracking.")
        return
    from src.config import BASE_DIR
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
    mlflow.set_experiment("Customer Churn Prediction")
    with mlflow.start_run(run_name=meta["model_version"]):
        mlflow.log_params({"model": meta["model_name"], "rows": meta["rows"],
                           "threshold": meta["threshold"], "data_source": meta["data_source"]})
        for name, res in meta["cv_results"].items():
            mlflow.log_metric(f"cv_roc_auc_{name.replace(' ', '_').lower()}", res["roc_auc_mean"])
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]:
            mlflow.log_metric(f"test_{key}", meta["test_metrics"][key])
        mlflow.log_artifact(str(model_dir / "churn_model.pkl"))
        mlflow.log_artifact(str(model_dir / "metadata.json"))
    print("Logged to MLflow (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()
    train(use_mlflow=not args.no_mlflow)
