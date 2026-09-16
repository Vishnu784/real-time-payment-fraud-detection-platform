from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    transaction_id: str = Field(min_length=3, max_length=64)
    amount: float = Field(gt=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home_km: float = Field(ge=0, le=50_000)
    merchant_risk_score: float = Field(ge=0, le=1)
    velocity_1h: int = Field(ge=0, le=1_000)
    failed_attempts_24h: int = Field(ge=0, le=1_000)
    is_foreign: bool
    card_present: bool
    account_age_days: int = Field(ge=0, le=50_000)


class RiskFactor(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    risk_factors: list[RiskFactor]


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: str
    amount: float
    fraud_probability: float
    is_fraud: bool
    model_version: str
    created_at: datetime


class ServiceMetrics(BaseModel):
    total_scored: int
    flagged_as_fraud: int
    flag_rate: float
    average_fraud_probability: float

