from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'fraud.db'}")
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "fraud_model.joblib"))
)

