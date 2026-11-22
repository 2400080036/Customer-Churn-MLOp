"""Run model monitoring: data drift on logged predictions + optional performance check.

    python src/monitor.py                          # drift only
    python src/monitor.py --labeled new_data.csv   # also check accuracy on newly labelled customers
Exit code 1 means an ALERT was raised (useful for scheduled jobs / CI).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH, PREDICTION_LOG, REPORT_DIR
from src.monitoring import drift_report, performance_report


def run(labeled_path=None, log_path=PREDICTION_LOG) -> dict:
    meta = json.loads(METADATA_PATH.read_text())
    recent = pd.read_csv(log_path).tail(1000) if Path(log_path).exists() else pd.DataFrame()
    report = {"model_version": meta["model_version"],
              "drift": drift_report(meta["drift_baseline"], recent)}
    if labeled_path:
        report["performance"] = performance_report(
            joblib.load(MODEL_PATH), meta, pd.read_csv(labeled_path))
    statuses = [report["drift"]["status"]] + [report.get("performance", {}).get("status", "OK")]
    report["overall"] = "ALERT" if "ALERT" in statuses else "WARN" if "WARN" in statuses else "OK"
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "monitoring.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", help="csv with the same columns as the training data")
    args = parser.parse_args()
    result = run(args.labeled)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["overall"] == "ALERT" else 0)
