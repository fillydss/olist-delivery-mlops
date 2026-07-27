"""
train.py — Olist delivery-time pipeline, Phase 2 (training + tracking).

Trains five regression models, logs params/metrics/models to MLflow, registers
the best by test MAE, AND saves the winning model to artifacts/model.joblib so
the prediction API can load it directly (no MLflow needed at serve time).

Run from the repo root:   python src/train.py
View results:             mlflow ui   (http://127.0.0.1:5000)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED = Path("data/processed")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TARGET = "delivery_days"
EXPERIMENT = "olist-delivery-time"
REGISTERED_MODEL = "olist-delivery-time-model"
SERIALIZATION = "cloudpickle"


# 1. Load the encoded train/test sets
def load_data():
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    return (train.drop(columns=[TARGET]), train[TARGET],
            test.drop(columns=[TARGET]), test[TARGET])


# 2. The five models to benchmark
def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=300, random_state=42,
        ),
    }


# 3. Evaluation helper
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }


# 4. Train, evaluate, log every model
def run():
    X_train, y_train, X_test, y_test = load_data()
    mlflow.set_experiment(EXPERIMENT)

    results = {}
    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_param("model", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model", serialization_format=SERIALIZATION)
            results[name] = metrics
            print(f"{name:22s}  MAE={metrics['mae']:.3f}  "
                  f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.4f}")

    return results, (X_train, y_train, X_test, y_test)


# 5. Pick the winner, register it, and save it as a plain file for serving
def register_best(results, data):
    X_train, y_train, X_test, y_test = data
    best_name = min(results, key=lambda n: results[n]["mae"])
    print(f"\nBest model by test MAE: {best_name} (MAE={results[best_name]['mae']:.3f})")

    best_model = get_models()[best_name]
    with mlflow.start_run(run_name=f"{best_name}-registered"):
        best_model.fit(X_train, y_train)
        mlflow.log_param("model", best_name)
        mlflow.log_metrics(results[best_name])
        mlflow.sklearn.log_model(
            best_model, name="model",
            serialization_format=SERIALIZATION,
            registered_model_name=REGISTERED_MODEL,
        )

    # Save the winner as a standalone file the API loads directly.
    joblib.dump(best_model, ARTIFACTS / "model.joblib")
    print(f"Registered '{best_name}' in MLflow and saved -> {ARTIFACTS/'model.joblib'}")


def main():
    results, data = run()
    register_best(results, data)


if __name__ == "__main__":
    main()
