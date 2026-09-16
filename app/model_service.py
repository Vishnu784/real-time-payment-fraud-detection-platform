from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.schemas import RiskFactor, TransactionInput

FEATURES = [
    "amount",
    "hour",
    "distance_from_home_km",
    "merchant_risk_score",
    "velocity_1h",
    "failed_attempts_24h",
    "is_foreign",
    "card_present",
    "account_age_days",
]


@dataclass(frozen=True)
class Prediction:
    probability: float
    is_fraud: bool
    threshold: float
    version: str
    risk_factors: list[RiskFactor]


class FraudModel:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run: python scripts/train.py"
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.version = str(bundle["version"])

    def predict(self, transaction: TransactionInput) -> Prediction:
        values = transaction.model_dump(include=set(FEATURES))
        frame = pd.DataFrame([values], columns=FEATURES)
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        return Prediction(
            probability=probability,
            is_fraud=probability >= self.threshold,
            threshold=self.threshold,
            version=self.version,
            risk_factors=self._explain(frame),
        )

    def _explain(self, frame: pd.DataFrame) -> list[RiskFactor]:
        scaler = self.pipeline.named_steps["scaler"]
        classifier = self.pipeline.named_steps["classifier"]
        scaled = scaler.transform(frame)[0]
        contributions = scaled * classifier.coef_[0]
        order = np.argsort(np.abs(contributions))[::-1][:3]
        return [
            RiskFactor(
                feature=FEATURES[index],
                impact=round(float(contributions[index]), 4),
                direction="increases_risk" if contributions[index] >= 0 else "decreases_risk",
            )
            for index in order
        ]

