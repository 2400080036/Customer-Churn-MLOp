import json

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL, FEATURES, METADATA_PATH, MODEL_PATH
from src.data import load_data, split, validate
from src.monitoring import drift_report, psi


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_dataset_is_large_enough_and_imbalanced(df):
    assert len(df) >= 5000
    rate = (df["churn"] == "Yes").mean()
    assert 0.15 < rate < 0.40          # realistic, not a perfect 50/50


def test_validation_rejects_bad_data(df):
    with pytest.raises(ValueError):
        validate(df.assign(contract="Lifetime"))
    with pytest.raises(ValueError):
        validate(df.drop(columns=["tenure"]))
    with pytest.raises(ValueError):
        validate(pd.concat([df, df.head(3)]))      # duplicate customer ids


def test_train_and_test_do_not_overlap(df):
    X_train, X_test, _, _ = split(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_test) == pytest.approx(0.2 * len(df), abs=2)


def test_split_keeps_churn_rate(df):
    _, _, y_train, y_test = split(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_saved_model_is_better_than_guessing_on_unseen_data(df):
    import joblib
    from sklearn.metrics import roc_auc_score
    _, X_test, _, y_test = split(df)
    auc = roc_auc_score(y_test, joblib.load(MODEL_PATH).predict_proba(X_test)[:, 1])
    assert auc > 0.70
    assert auc < 0.99                  # a near-perfect score would signal leakage


def test_probabilities_are_realistic(df):
    import joblib
    _, X_test, _, _ = split(df)
    p = joblib.load(MODEL_PATH).predict_proba(X_test)[:, 1]
    assert (p > 0.99).sum() == 0          # never claims near-certain churn
    assert (p < 0.01).mean() < 0.05       # very few near-certain 'stays'


def test_psi_is_zero_for_same_and_large_for_shifted_distribution():
    same = [0.1] * 10
    assert psi(same, same) == pytest.approx(0, abs=1e-9)
    assert psi(same, [0.5, 0.5] + [0.0] * 8) > 0.25


def test_drift_detection_flags_a_shifted_customer_base(df):
    meta = json.loads(METADATA_PATH.read_text())
    normal = df.sample(500, random_state=1)
    assert drift_report(meta["drift_baseline"], normal)["status"] == "OK"
    shifted = normal.assign(contract="Month-to-month", monthly_charges=110.0)
    report = drift_report(meta["drift_baseline"], shifted)
    assert report["status"] == "ALERT"
    assert report["features"]["contract"]["status"] == "ALERT"


def test_drift_needs_enough_rows(df):
    meta = json.loads(METADATA_PATH.read_text())
    assert drift_report(meta["drift_baseline"], df.head(10))["status"] == "NOT_ENOUGH_DATA"
