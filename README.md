# Real-Time Payment Fraud Detection Platform

An end-to-end portfolio project that trains an imbalanced fraud-classification model, serves real-time predictions through FastAPI, explains each score, and stores decisions for monitoring.

> The included data is synthetic and intended for education and demonstrations. This system must not be used to make real financial decisions without representative data, bias testing, security review, monitoring, and human oversight.

## What this demonstrates

- Imbalanced classification with class weighting
- Threshold tuning using the precision-recall curve
- ROC-AUC, PR-AUC, precision, recall, F1 and confusion-matrix evaluation
- Real-time REST prediction API with input validation
- Per-prediction risk-factor explanations
- SQLite locally and PostgreSQL through Docker Compose
- Transaction audit log and aggregate service metrics
- Automated API tests and GitHub Actions CI

## Architecture

```text
Client -> FastAPI /predict -> ML pipeline -> fraud probability + explanation
                       |                         |
                       +----> SQL audit log <---+
                                  |
                        /transactions, /metrics
```

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train.py
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Example prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn-demo-001",
    "amount": 8250,
    "hour": 2,
    "distance_from_home_km": 480,
    "merchant_risk_score": 0.91,
    "velocity_1h": 12,
    "failed_attempts_24h": 5,
    "is_foreign": true,
    "card_present": false,
    "account_age_days": 15
  }'
```

Other endpoints:

- `GET /health`
- `GET /transactions?limit=20`
- `GET /metrics`

## Run with PostgreSQL

```bash
docker compose up --build
```

## Test

```bash
pytest -q
```

## Honest resume wording

After you run, understand, and publish the project, adapt these bullets using the real values in `artifacts/metrics.json`:

- Developed an end-to-end payment fraud detection platform using Python, Scikit-learn, FastAPI and SQL, applying class-weighted classification and precision-recall threshold tuning to imbalanced transaction data.
- Built REST endpoints for real-time scoring, risk-factor explanations, transaction audit logging and service metrics; containerized the API with Docker and added automated tests and GitHub Actions CI.
- Evaluated model performance using PR-AUC, ROC-AUC, precision, recall, F1-score and a confusion matrix rather than relying on accuracy alone.

Do not present synthetic-data performance as production or bank performance. Label it as a portfolio demonstration during interviews.

