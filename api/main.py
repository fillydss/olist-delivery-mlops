"""
main.py — FastAPI service for Olist delivery-time prediction, Phase 3.

Loads the saved model + preprocessor once at startup, then exposes:
  GET  /health   -> liveness check
  POST /predict  -> delivery-time prediction for one order

Run from the repo root:  uvicorn api.main:app --reload
Interactive docs:        http://127.0.0.1:8000/docs
"""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import OrderRequest, PredictionResponse

ARTIFACTS = Path("artifacts")

app = FastAPI(title="Olist Delivery-Time Predictor", version="1.0")

# Load artifacts ONCE at import time, not per request (loading is expensive).
model = joblib.load(ARTIFACTS / "model.joblib")
pre = joblib.load(ARTIFACTS / "preprocessor.joblib")


def build_feature_row(order: OrderRequest) -> pd.DataFrame:
    """Reproduce the training-time feature engineering for a single order."""
    global_median = pre["global_median"]

    # Median-encode city and zip using the saved training maps (fallback = global median).
    city_med = pre["median_maps"]["customer_city"].get(order.customer_city, global_median)
    zip_med = pre["median_maps"]["customer_zip_code_prefix"].get(
        order.customer_zip_code_prefix, global_median
    )

    values = {
        "nb_items": order.nb_items,
        "n_sellers": order.n_sellers,
        "n_categories": order.n_categories,
        "total_price": order.total_price,
        "total_freight": order.total_freight,
        "total_weight_g": order.total_weight_g,
        "avg_length_cm": order.avg_length_cm,
        "avg_height_cm": order.avg_height_cm,
        "avg_width_cm": order.avg_width_cm,
        "purchase_hour": order.purchase_hour,
        "purchase_dow": order.purchase_dow,
        "customer_city_med": city_med,
        "customer_zip_code_prefix_med": zip_med,
        f"customer_state_{order.customer_state}": 1,  # one-hot the state
    }

    # Start from an all-zero row with the EXACT training columns, then fill in.
    row = pd.DataFrame(0, index=[0], columns=pre["feature_columns"])
    for col, val in values.items():
        if col in row.columns:      # unknown state -> stays all-zero, which is fine
            row.at[0, col] = val
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderRequest):
    row = build_feature_row(order)
    prediction = float(model.predict(row)[0])
    return PredictionResponse(predicted_delivery_days=round(prediction, 2))
