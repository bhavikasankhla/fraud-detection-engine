from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import time

# ── Load model artifacts once at startup ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "model", "feature_names.pkl")
THRESH_PATH= os.path.join(BASE_DIR, "model", "best_threshold.pkl")

try:
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_PATH)
    threshold     = joblib.load(THRESH_PATH)
    print(f"✅ Model loaded | Features: {len(feature_names)} | Threshold: {threshold:.3f}")
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time payment fraud detection using XGBoost. "
                "Trained on IEEE-CIS dataset (590K transactions, AUC-ROC 0.927).",
    version="1.0.0"
)

# ── Request schema ───────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, example=150.00,
                                  description="Transaction amount in USD")
    TransactionDT:  int   = Field(..., example=86400,
                                  description="Seconds offset from reference point")
    ProductCD:      Optional[str]   = Field(None, example="W")
    card1:          Optional[float] = Field(None, example=9500)
    card2:          Optional[float] = Field(None, example=325)
    card4:          Optional[str]   = Field(None, example="visa")
    card6:          Optional[str]   = Field(None, example="debit")
    P_emaildomain:  Optional[str]   = Field(None, example="gmail.com")
    R_emaildomain:  Optional[str]   = Field(None, example="yahoo.com")
    addr1:          Optional[float] = Field(None, example=299)
    addr2:          Optional[float] = Field(None, example=87)

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 150.00,
                "TransactionDT":  86400,
                "ProductCD":      "W",
                "card1":          9500,
                "card2":          325,
                "card4":          "visa",
                "P_emaildomain":  "gmail.com"
            }
        }

# ── Response schema ──────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    transaction_id:    str
    fraud_probability: float
    decision:          str
    risk_level:        str
    threshold_used:    float
    response_time_ms:  float
    model_version:     str = "xgboost-v1.0"

# ── Helper ───────────────────────────────────────────────────────────────
def _build_input(txn: TransactionRequest) -> pd.DataFrame:
    row = pd.DataFrame([txn.model_dump()])

    from src.features import engineer_features, align_features, encode_categoricals
    row = engineer_features(row)
    row = encode_categoricals(row)        # ← this line is the fix
    row = align_features(row, feature_names)
    return row

# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Check API is live and model is loaded."""
    return {
        "status":          "healthy",
        "model":           "XGBoost Fraud Detector",
        "features_loaded": len(feature_names),
        "threshold":       round(threshold, 3)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: TransactionRequest):
    """
    Submit a payment transaction and receive a real-time fraud prediction.

    Returns fraud probability (0–1), decision (FRAUD / LEGITIMATE),
    and risk level (HIGH / MEDIUM / LOW).
    """
    start = time.time()

    try:
        X   = _build_input(txn)
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    decision   = "FRAUD"      if prob >= threshold else "LEGITIMATE"
    risk_level = "HIGH"       if prob > 0.70 \
            else "MEDIUM"     if prob > 0.40 \
            else "LOW"

    return PredictionResponse(
        transaction_id    = f"txn_{int(time.time()*1000)}",
        fraud_probability = round(prob, 4),
        decision          = decision,
        risk_level        = risk_level,
        threshold_used    = round(float(threshold), 3),
        response_time_ms  = round((time.time() - start) * 1000, 2)
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionRequest]):
    """
    Submit up to 100 transactions and get predictions for all.
    Useful for testing the pipeline with multiple records.
    """
    if len(transactions) > 100:
        raise HTTPException(status_code=400,
                            detail="Batch size limited to 100 transactions")
    return [predict(txn) for txn in transactions]