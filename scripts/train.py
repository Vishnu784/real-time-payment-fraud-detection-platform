from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import FEATURES  # noqa: E402

RANDOM_SEED = 42


def generate_transactions(rows: int = 30_000) -> pd.DataFrame:
    """Generate reproducible, imbalanced demo data with realistic fraud signals."""
    rng = np.random.default_rng(RANDOM_SEED)
    frame = pd.DataFrame(
        {
            "amount": np.clip(rng.lognormal(4.0, 1.15, rows), 1, 25_000),
            "hour": rng.integers(0, 24, rows),
            "distance_from_home_km": np.clip(rng.exponential(18, rows), 0, 2_000),
            "merchant_risk_score": rng.beta(1.5, 8, rows),
            "velocity_1h": np.clip(rng.poisson(1.4, rows), 0, 30),
            "failed_attempts_24h": np.clip(rng.poisson(0.25, rows), 0, 20),
            "is_foreign": rng.binomial(1, 0.07, rows),
            "card_present": rng.binomial(1, 0.72, rows),
            "account_age_days": np.clip(rng.gamma(2.2, 420, rows), 0, 7_000).astype(int),
        }
    )

    night = ((frame["hour"] <= 4) | (frame["hour"] >= 23)).astype(float)
    logit = (
        -7.0
        + 0.0045 * frame["amount"]
        + 2.8 * frame["merchant_risk_score"]
        + 0.22 * frame["velocity_1h"]
        + 0.55 * frame["failed_attempts_24h"]
        + 1.15 * frame["is_foreign"]
        - 0.75 * frame["card_present"]
        + 0.006 * frame["distance_from_home_km"]
        + 0.65 * night
        - 0.0005 * frame["account_age_days"]
    )
    probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    frame["is_fraud"] = rng.binomial(1, probability)
    return frame


def best_f1_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def main() -> None:
    data = generate_transactions()
    train, test = train_test_split(
        data, test_size=0.25, stratify=data["is_fraud"], random_state=RANDOM_SEED
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1_000)),
        ]
    )
    pipeline.fit(train[FEATURES], train["is_fraud"])
    probability = pipeline.predict_proba(test[FEATURES])[:, 1]
    threshold = best_f1_threshold(test["is_fraud"], probability)
    prediction = (probability >= threshold).astype(int)

    report = {
        "rows": len(data),
        "fraud_rate": round(float(data["is_fraud"].mean()), 6),
        "threshold": round(threshold, 6),
        "roc_auc": round(float(roc_auc_score(test["is_fraud"], probability)), 6),
        "pr_auc": round(float(average_precision_score(test["is_fraud"], probability)), 6),
        "confusion_matrix": confusion_matrix(test["is_fraud"], prediction).tolist(),
        "classification_report": classification_report(
            test["is_fraud"], prediction, output_dict=True, zero_division=0
        ),
    }

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "threshold": threshold, "version": "1.0.0"},
        artifact_dir / "fraud_model.joblib",
    )
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()

