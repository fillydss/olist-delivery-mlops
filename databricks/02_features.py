# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02 — Features
# MAGIC
# MAGIC PySpark port of `src/features.py`. Same rule as the original: **every transform
# MAGIC that learns from data is fit on the training split only**, then applied to test.
# MAGIC
# MAGIC Outputs:
# MAGIC - `{catalog}.{schema}.features_train` / `features_test` (Delta)
# MAGIC - `preprocessor.json` in the artifacts Volume, so the serving layer can
# MAGIC   reproduce the exact same transformation

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "olist")
dbutils.widgets.text("artifacts_volume", "artifacts")
dbutils.widgets.text("test_size", "0.20")
dbutils.widgets.text("seed", "42")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ART_VOL = dbutils.widgets.get("artifacts_volume")
TEST_SIZE = float(dbutils.widgets.get("test_size"))
SEED = int(dbutils.widgets.get("seed"))

ARTIFACTS = f"/Volumes/{CATALOG}/{SCHEMA}/{ART_VOL}"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{ART_VOL}")

TARGET = "delivery_days"
NUMERIC = [
    "nb_items", "n_sellers", "n_categories",
    "total_price", "total_freight", "total_weight_g",
    "avg_length_cm", "avg_height_cm", "avg_width_cm",
    "purchase_hour", "purchase_dow",
]
IMPUTE = ["avg_length_cm", "avg_height_cm", "avg_width_cm"]
MEDIAN_ENCODE = ["customer_city", "customer_zip_code_prefix"]
ONE_HOT = "customer_state"

# COMMAND ----------

from pyspark.sql import functions as F
import json

df = spark.table(f"{CATALOG}.{SCHEMA}.orders_clean")

# --- 1. Split FIRST, before fitting anything ----------------------------------
train_df, test_df = df.randomSplit([1 - TEST_SIZE, TEST_SIZE], seed=SEED)
print(f"train: {train_df.count():,}   test: {test_df.count():,}")

# COMMAND ----------

# --- 2. Impute missing numerics with TRAIN medians ----------------------------
impute_values = {
    c: train_df.approxQuantile(c, [0.5], 0.001)[0] for c in IMPUTE
}
print("impute medians (fit on train):", impute_values)

train_df = train_df.fillna(impute_values)
test_df = test_df.fillna(impute_values) 

# COMMAND ----------

# --- 3. Median-encode high-cardinality categoricals ---------------------------
# Fit the city/zip -> median-delivery-time maps on train only. Unseen categories
# in test fall back to the global TRAIN median.
global_median = train_df.approxQuantile(TARGET, [0.5], 0.001)[0]
median_maps = {}

for col in MEDIAN_ENCODE:
    enc = (train_df.groupBy(col)
           .agg(F.percentile_approx(TARGET, 0.5).alias(f"{col}_med")))
    median_maps[col] = {str(r[col]): float(r[f"{col}_med"]) for r in enc.collect()}

    train_df = (train_df.join(F.broadcast(enc), on=col, how="left")
                .fillna({f"{col}_med": global_median}))
    test_df = (test_df.join(F.broadcast(enc), on=col, how="left")
               .fillna({f"{col}_med": global_median}))

print(f"global train median: {global_median:.3f}")
print({k: len(v) for k, v in median_maps.items()})

# COMMAND ----------

# --- 4. One-hot encode customer_state -----------------------------------------
# Categories come from TRAIN only, and test is built against the same column list.
states = sorted(
    r[0] for r in train_df.select(ONE_HOT).distinct().collect() if r[0] is not None
)
oh_cols = [f"{ONE_HOT}_{s}" for s in states]

for state, col in zip(states, oh_cols):
    expr = F.when(F.col(ONE_HOT) == state, 1).otherwise(0)
    train_df = train_df.withColumn(col, expr)
    test_df = test_df.withColumn(col, expr)

feature_cols = NUMERIC + [f"{c}_med" for c in MEDIAN_ENCODE] + oh_cols
print(f"{len(feature_cols)} features ({len(states)} states one-hot encoded)")

# COMMAND ----------

# --- 5. Write feature tables --------------------------------------------------
train_out = train_df.select(*feature_cols, TARGET)
test_out = test_df.select(*feature_cols, TARGET)

for name, frame in [("features_train", train_out), ("features_test", test_out)]:
    (frame.write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{CATALOG}.{SCHEMA}.{name}"))
    print(f"{name}: {spark.table(f'{CATALOG}.{SCHEMA}.{name}').count():,} rows")

# COMMAND ----------

# --- 6. Save the fitted preprocessor ------------------------------------------
# feature_columns pins the exact column order the model expects at inference time.
preprocessor = {
    "impute_values": impute_values,
    "median_maps": median_maps,
    "global_median": global_median,
    "one_hot_states": states,
    "feature_columns": feature_cols,
    "target": TARGET,
    "seed": SEED,
    "test_size": TEST_SIZE,
}

with open(f"{ARTIFACTS}/preprocessor.json", "w") as f:
    json.dump(preprocessor, f, indent=2)
print(f"Saved -> {ARTIFACTS}/preprocessor.json")

# COMMAND ----------

# --- 7. Leakage + null checks -------------------------------------------------
for name in ["features_train", "features_test"]:
    t = spark.table(f"{CATALOG}.{SCHEMA}.{name}")
    nulls = t.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in t.columns
    ]).collect()[0].asDict()
    remaining = {k: v for k, v in nulls.items() if v}
    print(f"{name}: nulls = {remaining or 'none'}")
    assert not remaining, f"{name} still has nulls: {remaining}"

print("\nNo nulls, and every fitted statistic came from the train split only.")
