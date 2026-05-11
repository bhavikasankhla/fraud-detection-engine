# Real-Time Fraud Detection Engine

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.927-brightgreen)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

End-to-end machine learning system for real-time payment fraud detection.  
Trained on the IEEE-CIS Fraud Detection dataset (590K transactions).  
Served via a production-ready FastAPI REST API with Docker support.

---

## Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | **0.927** |
| F1 Score (fraud class) | 0.632 |
| Precision (fraud) | 0.78 |
| Fraud recall | 0.53 |
| Features engineered | 442 |
| Training samples | 472,432 |

---

## Project Architecture

```
Data (IEEE-CIS Kaggle) 
  → EDA & Feature Engineering (Pandas, NumPy)
  → SMOTE oversampling (imbalanced-learn)
  → XGBoost training (500 trees, AUC-ROC 0.927)
  → Model serialisation (joblib)
  → FastAPI REST API (/predict, /predict/batch, /health)
  → Docker containerisation
```

---

## Folder Structure

```
fraud-detection-engine/
├── api/
│   └── main.py          # FastAPI app — /predict, /batch, /health
├── src/
│   └── features.py      # Feature engineering pipeline
├── model/
│   ├── fraud_model.pkl
│   ├── feature_names.pkl
│   └── best_threshold.pkl
├── notebooks/
│   └── fraud_detection.ipynb   # Full EDA + training notebook (Kaggle)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Run locally
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-engine
cd fraud-detection-engine
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Run with Docker
```bash
docker-compose up --build
```

API will be live at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## API Usage

### POST /predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 2500.00,
    "TransactionDT": 3600,
    "card1": 9500,
    "P_emaildomain": "gmail.com"
  }'
```

**Response:**
```json
{
  "transaction_id": "txn_1778504200494",
  "fraud_probability": 0.3049,
  "decision": "LEGITIMATE",
  "risk_level": "LOW",
  "threshold_used": 0.318,
  "response_time_ms": 390.85,
  "model_version": "xgboost-v1.0"
}
```

### GET /health
```bash
curl http://localhost:8000/health
```

---

## Key Engineering Decisions

**Why XGBoost over Random Forest?**  
XGBoost's gradient boosting handles the sparse V-features (300+ anonymised columns) significantly better. AUC improved from 0.83 (RF) to 0.927 (XGBoost).

**Why SMOTE at 0.3 ratio, not 1.0?**  
Full 1:1 oversampling caused overfitting on synthetic samples. A 0.3 ratio (23% fraud after SMOTE) gave the best validation AUC without memorising synthetic patterns.

**Why threshold 0.318 instead of 0.5?**  
Default 0.5 threshold missed too many fraud cases. Optimised threshold was chosen by maximising F1 on the precision-recall curve — standard practice in fraud/medical ML.

---

## Dataset
https://www.kaggle.com/code/bhavikasankhla/fraud-detection

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)  
590,540 transactions · 3.5% fraud rate · 394 raw features

---

## Tech Stack

`Python` `XGBoost` `Scikit-learn` `Pandas` `NumPy`  
`FastAPI` `Uvicorn` `Docker` `joblib` `imbalanced-learn`
