from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["DATABASE_URL"] = f"sqlite:///{PROJECT_ROOT / 'test_fraud.db'}"
os.environ["MODEL_PATH"] = str(PROJECT_ROOT / "artifacts" / "fraud_model.joblib")


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    model_path = PROJECT_ROOT / "artifacts" / "fraud_model.joblib"
    if not model_path.exists():
        subprocess.run([sys.executable, "scripts/train.py"], cwd=PROJECT_ROOT, check=True)
    yield
    test_db = PROJECT_ROOT / "test_fraud.db"
    if test_db.exists():
        test_db.unlink()
