from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    amount: Mapped[float] = mapped_column(Float)
    hour: Mapped[int] = mapped_column(Integer)
    distance_from_home_km: Mapped[float] = mapped_column(Float)
    merchant_risk_score: Mapped[float] = mapped_column(Float)
    velocity_1h: Mapped[int] = mapped_column(Integer)
    failed_attempts_24h: Mapped[int] = mapped_column(Integer)
    is_foreign: Mapped[bool] = mapped_column(Boolean)
    card_present: Mapped[bool] = mapped_column(Boolean)
    account_age_days: Mapped[int] = mapped_column(Integer)
    fraud_probability: Mapped[float] = mapped_column(Float)
    is_fraud: Mapped[bool] = mapped_column(Boolean, index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

