# Databricks port

The local pipeline (`src/`) rewritten for Databricks: PySpark transformations over
Delta tables in Unity Catalog, managed MLflow tracking, a UC-registered model, and
a scheduled Workflow that only retrains when the drift monitor says so.

Runs on **Databricks Free Edition** (serverless, no cloud account, no card).

```
Volume: raw CSVs
   |
   |-- 01_ingest    Spark joins + target       -->  orders_clean        (Delta)
   |-- 02_features  leakage-safe encoding      -->  features_train/test (Delta)
   |                                            -->  preprocessor.json   (Volume)
   |-- 04_monitor   Spark KS drift test        -->  drift_reports       (Delta)
   |                                            -->  taskValue: retrain_triggered
   '-- 03_train     5-model benchmark + MLflow -->  delivery_time_model@champion (UC)
                    (runs only when drift is flagged)
```

## Setup

1. **Create the workspace**, sign up at
   `databricks.com/learn/free-edition`. No credit card, no cloud account.
   (Free Edition replaced Community Edition, which was retired on 1 Jan 2026.)

2. **Create the schema and volumes** in a SQL editor or notebook cell:
   ```sql
   CREATE SCHEMA IF NOT EXISTS workspace.olist;
   CREATE VOLUME IF NOT EXISTS workspace.olist.raw;
   CREATE VOLUME IF NOT EXISTS workspace.olist.artifacts;
   ```

3. **Upload the data**, Catalog → `workspace` → `olist` → `raw` → Upload, and drop
   in the four Olist CSVs: `olist_orders_dataset.csv`,
   `olist_order_items_dataset.csv`, `olist_products_dataset.csv`,
   `olist_customers_dataset.csv`.

4. **Import the notebooks**, Workspace → Import → File, and select the four `.py`
   files. They are in Databricks source format, so they import as notebooks, not
   as scripts. (Better: connect the repo via Git folders so the notebooks stay
   version-controlled, that is also what the job JSON's paths assume.)

5. **Run them in order**, `01` → `02` → `03` → `04`. Each is idempotent; rerun
   freely. Everything attaches to serverless automatically.

6. **Create the Workflow**, Jobs & Pipelines → Create job. Build the five tasks in
   the UI to match `resources/olist_pipeline_job.json` (it is worth doing once by
   hand to see the DAG editor), or paste the JSON via *Edit as JSON*. Set the
   failure-notification email before saving.

## What changed from the local version, and why

| Local (`src/`) | Databricks | Reason |
|---|---|---|
| `pd.read_csv` | `spark.read.csv` from a UC Volume | Volumes are governed storage; no laptop paths |
| CSVs in `data/processed/` | Delta tables in Unity Catalog | ACID writes, schema enforcement, time travel, queryable from SQL |
| `train_test_split(random_state=42)` | `randomSplit(seed=42)` | Deterministic in Spark, but **not** the same rows as sklearn; metrics move slightly |
| `.median()` | `approxQuantile(..., 0.001)` | Exact quantiles need a full sort per partition |
| `groupby().median()` map | `percentile_approx` + broadcast join | A `.map()` on a Spark column does not exist; the encoder becomes a join |
| `pd.get_dummies` + `reindex` | explicit `when/otherwise` per train category | Reproduces the reindex guarantee: test can never introduce a column |
| local MLflow + `mlflow ui` | managed MLflow + UC registry | No server to run; models get three-level names and aliases |
| `joblib` preprocessor | `preprocessor.json` in a Volume | Readable, versionable, not pinned to a sklearn version |
| `scipy.stats.ks_2samp` | KS statistic via Spark window function | scipy needs both samples in memory; this scales |
| `python src/x.py` by hand | Workflow DAG + weekly schedule | Dependencies, retries, task values, conditional retrain |

## Gotchas worth knowing (they came up during the port)

- **`sum()` of an all-null group.** pandas returns `0.0`, Spark returns `NULL`.
  Orders where every product had a missing weight silently turned `total_weight_g`
  into a null column. Caught by diffing the Spark output against the pandas output
  row-by-row; fixed with `coalesce(sum(...), lit(0.0))`. `countDistinct` and `avg`
  already match pandas' `nunique`/`mean`.
- **`dayofweek()` is 1-indexed from Sunday**, pandas' `dayofweek` is 0-indexed from
  Monday. `(dayofweek + 5) % 7` converts.
- **KS ties.** The CDF window must use `rangeBetween`, not `rowsBetween`, so all
  rows sharing a value are counted together. With `rowsBetween` the statistic is
  wrong on discrete features like `nb_items` and `purchase_dow`.
- **`randomSplit` is not `train_test_split`.** It partitions on a row hash, so the
  same seed gives a different (but stable) split. Expect the MAE to shift in the
  third decimal versus the local run. Say this rather than quietly reporting
  different numbers for the same project.
- **Unity Catalog registry needs a signature.** `mlflow.sklearn.log_model` without
  `signature=` is rejected. `infer_signature` handles it.
- **UC uses aliases, not stages.** `Staging`/`Production` are gone; load
  `models:/catalog.schema.model@champion`.

## Honest scope note

96K rows is not big data. Spark is not required here and a single pandas process is
faster for this volume. The port is about the platform mechanics, Delta, Unity
Catalog, managed MLflow, Workflows, and writing transformations that would still
work at 1000x, not about a performance win at this size.

