# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03 — Train + register
# MAGIC
# MAGIC Port of `src/train.py` to **managed MLflow** and the **Unity Catalog model
# MAGIC registry**. Same selection rule (lowest test MAE).
# MAGIC
# MAGIC The feature tables are ~96K rows and ~19 columns, which fits comfortably in
# MAGIC driver memory, so the models train single-node on pandas rather than in Spark
# MAGIC MLlib. That is a design choice, distributing a model this
# MAGIC size would add shuffle overhead and cost accuracy (MLlib has no XGBoost-equivalent
# MAGIC out of the box). Spark earns its keep in notebooks 01/02/04, where the work is
# MAGIC joins, aggregations and full-table scans.
# MAGIC
# MAGIC Output: `{catalog}.{schema}.delivery_time_model` with a `@champion` alias.

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost availability
# MAGIC
# MAGIC Free Edition restricts outbound network access to a limited set of trusted
# MAGIC domains, so `%pip install` is not guaranteed to work. Run the cell below: if
# MAGIC xgboost is already in the serverless image, nothing happens. If it is missing
# MAGIC and cannot be installed, the benchmark falls back to four models.
# MAGIC
# MAGIC That is not a meaningful loss — HistGradientBoosting landed within 0.03 days
# MAGIC MAE of XGBoost locally, and it was the winning model in the thesis. Say which
# MAGIC models ran; do not quietly report a four-model benchmark as five.

# COMMAND ----------

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
    print("xgboost already available")
except ImportError:
    HAS_XGB = False
    print("xgboost not in the base image — try the %pip cell below, then re-run")

# COMMAND ----------

# Only run this cell if the check above said xgboost is missing. %restart_python
# wipes the Python state, so re-run the notebook from the top afterwards.
# If the install fails on a network error, just carry on: the benchmark handles it.

%pip install xgboost --quiet
%restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "olist")
dbutils.widgets.text("experiment_path", "/Shared/olist-delivery-time")
dbutils.widgets.text("model_name", "delivery_time_model")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")
REGISTERED_MODEL = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('model_name')}"
TARGET = "delivery_days"

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Running the 4-model benchmark without XGBoost.")

# Point the registry at Unity Catalog rather than the legacy workspace registry.
# UC models are three-level names (catalog.schema.model) and use ALIASES instead
# of the old Staging/Production stages.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# Autologging would fire on every .fit() including the refit below, which
# clutters the experiment; logging explicitly keeps one run per model.
mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# --- 1. Load the feature tables -----------------------------------------------
train = spark.table(f"{CATALOG}.{SCHEMA}.features_train").toPandas()
test = spark.table(f"{CATALOG}.{SCHEMA}.features_test").toPandas()

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

# Column order must match what the preprocessor pinned in notebook 02.
X_test = X_test[X_train.columns]
print(f"train: {X_train.shape}   test: {X_test.shape}")

# COMMAND ----------

# --- 2. The five models -------------------------------------------------------
def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=300, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    return models


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "r2": float(r2_score(y, preds)),
    }

# COMMAND ----------

# --- 3. Benchmark, logging each model as a nested MLflow run ------------------
results = {}

with mlflow.start_run(run_name="benchmark") as parent:
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))
    mlflow.log_param("n_features", X_train.shape[1])

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            results[name] = metrics
            print(f"{name:22s} MAE={metrics['mae']:.3f} "
                  f"RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.4f}")

    mlflow.log_dict(results, "benchmark_results.json")

display(pd.DataFrame(results).T.sort_values("mae"))

# COMMAND ----------

# --- 4. Register the winner in Unity Catalog ----------------------------------
best_name = min(results, key=lambda n: results[n]["mae"])
print(f"Best by test MAE: {best_name} ({results[best_name]['mae']:.3f} days)")

best_model = get_models()[best_name]
best_model.fit(X_train, y_train)

# UC requires a model signature. infer_signature records input/output schema so
# the serving endpoint rejects malformed requests instead of silently coercing.
signature = infer_signature(X_train, best_model.predict(X_train[:5]))

with mlflow.start_run(run_name=f"{best_name}-champion") as run:
    mlflow.log_param("model", best_name)
    mlflow.log_metrics(results[best_name])
    info = mlflow.sklearn.log_model(
        best_model,
        name="model",                      # MLflow 3 arg; older versions use artifact_path=
        signature=signature,
        input_example=X_train.head(3),
        registered_model_name=REGISTERED_MODEL,
    )

print(f"Registered {REGISTERED_MODEL} version {info.registered_model_version}")

# COMMAND ----------

# --- 5. Alias the new version as @champion ------------------------------------
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    REGISTERED_MODEL, "champion", info.registered_model_version
)
client.set_model_version_tag(
    REGISTERED_MODEL, info.registered_model_version, "test_mae",
    f"{results[best_name]['mae']:.4f}"
)
print(f"{REGISTERED_MODEL}@champion -> v{info.registered_model_version}")

# COMMAND ----------

# --- 6. Verify the registered model loads and predicts ------------------------
loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
preds = loaded.predict(X_test.head(5))
print("sample predictions (days):", np.round(preds, 2))
print("actuals:                  ", np.round(y_test.head(5).values, 2))

# COMMAND ----------

dbutils.jobs.taskValues.set("best_model", best_name)
dbutils.jobs.taskValues.set("test_mae", results[best_name]["mae"])
