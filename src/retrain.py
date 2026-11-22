"""Retrain when monitoring raises an alert (or with --force) and promote the new model
only if it is not worse than the current one (champion / challenger).
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import METADATA_PATH, MODEL_DIR
from src.monitor import run as run_monitor
from src.train import train

TOLERANCE = 0.005   # new model may be at most this much worse in ROC-AUC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even without an alert")
    parser.add_argument("--labeled", help="newly labelled csv for the performance check")
    args = parser.parse_args()

    report = run_monitor(args.labeled)
    print("Monitoring status:", report["overall"])
    if report["overall"] != "ALERT" and not args.force:
        print("No alert - keeping the current model.")
        sys.exit(0)

    current = json.loads(METADATA_PATH.read_text())
    candidate_dir = MODEL_DIR / "candidate"
    new = train(model_dir=candidate_dir, use_mlflow=True)
    old_auc, new_auc = current["test_metrics"]["roc_auc"], new["test_metrics"]["roc_auc"]
    print(f"Current ROC-AUC {old_auc:.3f} | candidate {new_auc:.3f}")

    if new_auc >= old_auc - TOLERANCE:
        archive = MODEL_DIR / "archive" / f"{current['model_version']}_{datetime.now():%H%M%S}"
        archive.mkdir(parents=True)
        for name in ["churn_model.pkl", "metadata.json"]:
            shutil.move(str(MODEL_DIR / name), archive / name)
            shutil.copy(candidate_dir / name, MODEL_DIR / name)
        print(f"Promoted new model {new['model_version']} (old one archived).")
    else:
        print("Candidate is worse - keeping the current model.")
    shutil.rmtree(candidate_dir, ignore_errors=True)
