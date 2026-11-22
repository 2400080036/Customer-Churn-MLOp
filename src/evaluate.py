"""Evaluate the saved model on the held-out TEST data only (never on training data)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, roc_auc_score,
                             roc_curve)

from src.config import METADATA_PATH, MODEL_PATH, REPORT_DIR
from src.data import load_data, split

if __name__ == "__main__":
    meta = json.loads(METADATA_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = split(load_data())        # same split as training
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= meta["threshold"]).astype(int)

    print(f"Held-out test set: {len(y_test)} customers (never used for training)")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}   (95% CI {meta['test_metrics']['roc_auc_ci95'][0]:.3f}"
          f" - {meta['test_metrics']['roc_auc_ci95'][1]:.3f})")
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))

    REPORT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax[0].plot(fpr, tpr, label=f"Model (AUC {roc_auc_score(y_test, proba):.2f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Churn"],
                                            ax=ax[1], colorbar=False)
    ax[1].set_title(f"Confusion matrix (threshold {meta['threshold']:.2f})")
    frac, mean = calibration_curve(y_test, proba, n_bins=8)
    ax[2].plot(mean, frac, marker="o", label="Model")
    ax[2].plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    ax[2].set(title="Probability calibration", xlabel="Predicted probability", ylabel="Actual churn rate")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "evaluation.png", dpi=130)

    imp = meta["permutation_importance"]
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.barh(list(imp)[::-1], list(imp.values())[::-1], color="#1E3A5F")
    ax2.set(title="Feature importance (drop in AUC when shuffled)", xlabel="Importance")
    fig2.tight_layout()
    fig2.savefig(REPORT_DIR / "feature_importance.png", dpi=130)

    (REPORT_DIR / "evaluation.json").write_text(json.dumps(
        {"model_version": meta["model_version"], **meta["test_metrics"]}, indent=2))
    print(f"Saved charts and evaluation.json to {REPORT_DIR}")
