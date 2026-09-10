# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 04 — Drift monitor
# MAGIC
# MAGIC Port of `src/monitor.py`. Compares an earlier reference window against a later
# MAGIC current window with a two-sample Kolmogorov–Smirnov test per feature.
# MAGIC
# MAGIC The original called `scipy.stats.ks_2samp`, which needs both samples in memory.
# MAGIC Here the KS **statistic** is computed in Spark with a window function, the max
# MAGIC gap between the two empirical CDFs, so it scales past what fits on the driver.
# MAGIC Only the p-value is computed locally, from the statistic and the two sample
# MAGIC sizes. Matches `scipy.stats.ks_2samp` to floating point.
# MAGIC
# MAGIC Outputs: `{catalog}.{schema}.drift_reports` (Delta, one row per feature per run)
# MAGIC and a `retrain_triggered` task value the Workflow branches on.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "olist")
dbutils.widgets.text("reference_frac", "0.70")
dbutils.widgets.text("p_threshold", "0.05")
dbutils.widgets.text("drift_share_trigger", "0.5")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
REFERENCE_FRAC = float(dbutils.widgets.get("reference_frac"))
P_THRESHOLD = float(dbutils.widgets.get("p_threshold"))
DRIFT_SHARE_TRIGGER = float(dbutils.widgets.get("drift_share_trigger"))

TARGET = "delivery_days"
MONITOR_COLS = [
    "nb_items", "n_sellers", "n_categories",
    "total_price", "total_freight", "total_weight_g",
    "avg_length_cm", "avg_height_cm", "avg_width_cm",
    "purchase_hour", "purchase_dow",
    TARGET,
]

# COMMAND ----------

import math
from datetime import datetime
from pyspark.sql import functions as F, Window

df = spark.table(f"{CATALOG}.{SCHEMA}.orders_clean")
total = df.count()
cutoff = int(total * REFERENCE_FRAC)

# --- 1. Split by time ---------------------------------------------------------
# window = 0 -> reference (earlier orders, roughly what the model knows)
# window = 1 -> current   (later orders, "new incoming data")
tagged = (df
          .withColumn("_rn", F.row_number().over(
              Window.orderBy("order_purchase_timestamp")))
          .withColumn("_window", F.when(F.col("_rn") <= cutoff, 0).otherwise(1)))

n_ref = tagged.filter(F.col("_window") == 0).count()
n_cur = tagged.filter(F.col("_window") == 1).count()
print(f"reference: {n_ref:,} orders   current: {n_cur:,} orders")

# COMMAND ----------

# --- 2. Two-sample KS in Spark ------------------------------------------------
def spark_ks(frame, feature, n1, n2):
    d = (frame.select(F.col(feature).cast("double").alias("v"), "_window")
         .na.drop(subset=["v"]))

    w = Window.orderBy("v").rangeBetween(Window.unboundedPreceding, Window.currentRow)
    D = (d
         .withColumn("c_ref", F.sum(F.when(F.col("_window") == 0, 1).otherwise(0)).over(w))
         .withColumn("c_cur", F.sum(F.when(F.col("_window") == 1, 1).otherwise(0)).over(w))
         .withColumn("gap", F.abs(F.col("c_ref") / F.lit(n1) - F.col("c_cur") / F.lit(n2)))
         .agg(F.max("gap")).collect()[0][0])

    # Asymptotic Kolmogorov distribution, same closed form scipy uses in
    # method="asymp": Q(lambda) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lambda^2)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return float(D), float(min(max(p, 0.0), 1.0))

# COMMAND ----------

# --- 3. Run the test per feature ----------------------------------------------
run_ts = datetime.utcnow()
rows = []

for col in MONITOR_COLS:
    D, p = spark_ks(tagged, col, n_ref, n_cur)
    rows.append({
        "run_timestamp": run_ts,
        "feature": col,
        "ks_statistic": round(D, 6),
        "p_value": round(p, 8),
        "drifted": bool(p < P_THRESHOLD),
        "n_reference": n_ref,
        "n_current": n_cur,
    })
    print(f"{col:<20} D={D:.4f}  p={p:.3e}  {'DRIFT' if p < P_THRESHOLD else 'ok'}")

# COMMAND ----------

# --- 4. Verdict ---------------------------------------------------------------
n_drifted = sum(r["drifted"] for r in rows)
drift_share = n_drifted / len(rows)
trigger = drift_share > DRIFT_SHARE_TRIGGER

print(f"\nFeatures drifted: {n_drifted}/{len(rows)} ({drift_share:.0%})")
print(">>> RETRAIN TRIGGERED" if trigger else ">>> No retrain needed")

# COMMAND ----------

# --- 5. Append to the drift history table -------------------------------------
report = (spark.createDataFrame(rows)
          .withColumn("drift_share", F.lit(drift_share))
          .withColumn("retrain_triggered", F.lit(trigger)))

(report.write.mode("append")
 .option("mergeSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.drift_reports"))

display(spark.table(f"{CATALOG}.{SCHEMA}.drift_reports")
        .filter(F.col("run_timestamp") == F.lit(run_ts))
        .orderBy("p_value"))

# COMMAND ----------

# --- 6. Hand the verdict to the Workflow --------------------------------------
dbutils.jobs.taskValues.set("retrain_triggered", str(trigger).lower())
dbutils.jobs.taskValues.set("drift_share", drift_share)