from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import Integer, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import MODEL_PATH
from app.database import SessionLocal, TransactionRecord, init_db
from app.model_service import FraudModel
from app.schemas import (
    PredictionResponse,
    ServiceMetrics,
    TransactionInput,
    TransactionResponse,
)

model: FraudModel | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    init_db()
    model = FraudModel(MODEL_PATH)
    yield


app = FastAPI(
    title="Real-Time Payment Fraud Detection API",
    version="1.0.0",
    description="Scores payment transactions and records model decisions for monitoring.",
    lifespan=lifespan,
)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "fraud-detection-api", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy" if model is not None else "starting",
        "model_version": model.version if model else "unavailable",
    }


@app.post("/predict", response_model=PredictionResponse, status_code=201)
def predict(transaction: TransactionInput, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    prediction = model.predict(transaction)
    record = TransactionRecord(
        **transaction.model_dump(),
        fraud_probability=prediction.probability,
        is_fraud=prediction.is_fraud,
        model_version=prediction.version,
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="transaction_id already exists") from exc

    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=round(prediction.probability, 6),
        is_fraud=prediction.is_fraud,
        threshold=round(prediction.threshold, 6),
        model_version=prediction.version,
        risk_factors=prediction.risk_factors,
    )


@app.get("/transactions", response_model=list[TransactionResponse])
def transactions(
    limit: int = Query(default=20, ge=1, le=100), db: Session = Depends(get_db)
):
    statement = select(TransactionRecord).order_by(TransactionRecord.created_at.desc()).limit(limit)
    return list(db.scalars(statement))


@app.get("/metrics", response_model=ServiceMetrics)
def metrics(db: Session = Depends(get_db)):
    total, flagged, average = db.execute(
        select(
            func.count(TransactionRecord.id),
            func.sum(func.cast(TransactionRecord.is_fraud, Integer)),
            func.avg(TransactionRecord.fraud_probability),
        )
    ).one()
    total = int(total or 0)
    flagged = int(flagged or 0)
    return ServiceMetrics(
        total_scored=total,
        flagged_as_fraud=flagged,
        flag_rate=round(flagged / total, 6) if total else 0.0,
        average_fraud_probability=round(float(average or 0.0), 6),
    )
