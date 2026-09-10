# Databricks runbook, start to finish

Everything from "no account" to "screenshots in the README." Budget one evening for
Phases 0-6, and a second short session for Phases 7-9.

The Databricks UI moves around between releases. Where a menu name is likely to
drift I say what you're looking for, not just where it was, if a label doesn't
match, search the top bar (`Cmd/Ctrl + P` opens the command palette).

---

## Phase 0, Before you open Databricks (10 min)

Download the Olist dataset from Kaggle ("Brazilian E-Commerce Public Dataset by
Olist") and unzip it. You only need four of the nine CSVs:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_customers_dataset.csv`

Put them somewhere you can find in a file picker. Total ~50 MB.

---

## Phase 1, Account (5 min)

1. Go to **databricks.com/learn/free-edition** → *Sign up for Free Edition*.
2. Sign in with Google or Microsoft, or use email OTP. Those three are the only
   options on Free Edition, no SSO.
3. No credit card, no cloud account. A workspace is provisioned automatically.
4. **Do the LinkedIn verification if you're offered it.** It's in the account
   settings and unlocks higher limits, including outbound internet access, which
   is what makes `%pip install` work. Worth two minutes.

**What you're getting:** one workspace, one metastore, serverless compute only.
No cluster configuration screens, because there are no clusters to configure.

---

## Phase 2, Orientation (10 min, don't skip)

Spend ten minutes clicking around before you run anything. You need to be able to
describe this platform, not just have run code on it. The left sidebar:

| Nav item | What it is | Why you care |
|---|---|---|
| **Workspace** | Notebook/file tree | Where your four notebooks will live |
| **Catalog** | Unity Catalog browser | Catalogs → schemas → tables, volumes, models. This *is* the governance layer |
| **Jobs & Pipelines** | Workflows | Your DAG, schedule, run history |
| **Compute** | Serverless only here | Note what's *missing* vs paid, that's the Free Edition tradeoff |
| **Experiments** | Managed MLflow | Runs, metrics, artifacts |
| **Models** | UC model registry | Versions and aliases live here |
| **SQL Editor** | Warehouse queries | Your Delta tables are queryable SQL tables |

The one concept to actually understand: **Unity Catalog uses three-level names**,
`catalog.schema.object`. Your tables will be `workspace.olist.orders_clean`, your
model `workspace.olist.delivery_time_model`. That three-level naming is the single
most common Databricks interview question after "what is Delta Lake."

---

## Phase 3, Create the schema and volumes (5 min)

Open a new notebook: **Workspace** → your home folder → *Create* → *Notebook*. It
attaches to serverless automatically. Set the language to SQL for this cell (or
keep Python and wrap each line in `spark.sql(...)`).

```sql
CREATE SCHEMA IF NOT EXISTS workspace.olist;
CREATE VOLUME IF NOT EXISTS workspace.olist.raw;
CREATE VOLUME IF NOT EXISTS workspace.olist.artifacts;
```

Run it (`Shift + Enter`). Then go to **Catalog** and confirm you can see
`workspace` → `olist` → *Volumes* → `raw`, `artifacts`.

**What a Volume is:** governed file storage inside Unity Catalog, for things that
aren't tables, CSVs, model artifacts, images. It gets a path like
`/Volumes/workspace/olist/raw/` that both Spark and plain Python can read. This is
the modern replacement for DBFS, and knowing that distinction is worth a sentence
in an interview.

---

## Phase 4, Upload the data (5 min)

1. **Catalog** → `workspace` → `olist` → **Volumes** → `raw`.
2. Click **Upload to this volume** (top right).
3. Drag in all four CSVs at once. Wait for them to finish.
4. Verify from a notebook cell:
   ```python
   display(dbutils.fs.ls("/Volumes/workspace/olist/raw"))
   ```
   You should see four files.

**Don't** use the "Create table" wizard on the upload screen. Notebook 01 does the
ingestion deliberately, that's the work you want to be able to talk about.

---

## Phase 5, Import the notebooks (5 min)

**Option A, quick (do this if you just want to run it tonight):**

Workspace → your folder → *Create* → **Import** → *File* → select all four `.py`
files → Import. They're in Databricks source format, so they arrive as notebooks
with cells intact, not as flat scripts.

**Option B, better, and what the job JSON assumes:**

Workspace → *Create* → **Git folder** → paste
`https://github.com/fillydss/olist-delivery-mlops` → Create. Now the notebooks are
version-controlled, you can commit from the Databricks UI, and the job points at a
real repo path. Do this once you've committed the `databricks/` folder to GitHub.

Option B is the more credible story, "notebooks in a Git folder, job runs from the
repo" is how teams actually work, and it kills the "notebooks are unversioned mess"
objection to notebook-based pipelines.

---

## Phase 6, Run the pipeline (45-90 min, mostly waiting)

Run them **in order**, top to bottom, one notebook at a time. Use *Run all* only
after the first pass, for the first run, step through cell by cell so you see what
each one does.

### 01_ingest
- First cell creates the widgets. If you edit widget values, they persist.
- Expect ~96K rows out. Check the `DESCRIBE HISTORY` cell at the end, that's Delta
  time travel, and it's a good screenshot.
- **Screenshot:** the summary stats table and the history output.

### 02_features
- Watch the printed impute medians and the "global train median" line. Those numbers
  are proof the encoders were fit on train only.
- The assertion at the end fails loudly if any nulls survived. If it fails, something
  upstream changed, don't skip past it.
- **Screenshot:** the feature count line and the null check.

### 03_train
- Run the xgboost availability check cell first. If it says xgboost is missing, run
  the `%pip` cell, then **re-run the notebook from the top** (`%restart_python` wipes
  Python state). If the install fails on a network error, carry on, you'll get a
  four-model benchmark, which is fine as long as you say so.
- This is the slow one. RandomForest with 200 trees on serverless will take a few
  minutes.
- **Screenshot:** the sorted results dataframe, and the Experiments page showing the
  nested runs under `benchmark`.
- Then go to **Models** in the sidebar → `workspace.olist.delivery_time_model` →
  confirm version 1 exists with the `champion` alias. **Screenshot that page.** It's
  the single most convincing image in the set.

### 04_monitor
- Prints one line per feature. Locally, 9 of 12 drifted, so expect the retrain
  trigger to fire.
- **Screenshot:** the drift table from the `display()` cell.

**If a notebook fails partway:** all four are idempotent. Fix and re-run the whole
thing; nothing gets corrupted.

---

## Phase 7, Build the Workflow (20 min)

This is the part that turns "I ran some notebooks" into "I built a pipeline." Do it
in the UI the first time so you see the DAG editor.

1. **Jobs & Pipelines** → *Create* → **Job**.
2. Name it `olist-delivery-pipeline`.
3. Add task 1: name `ingest`, type *Notebook*, select `01_ingest`. Under
   **Parameters** add `catalog=workspace`, `schema=olist`, `volume=raw`.
4. Add task 2: `features` → `02_features`, **Depends on:** `ingest`.
5. Add task 3: `monitor` → `04_monitor`, **Depends on:** `features`.
6. Add task 4: `check_drift`, type **If/else condition**.
   - Condition: `{{tasks.monitor.values.retrain_triggered}}` **equals** `true`
   - Depends on: `monitor`
7. Add task 5: `train` → `03_train`, **Depends on:** `check_drift` with outcome
   **true**. Set retries to 1.
8. **Schedule:** weekly, Monday 06:00, Europe/Amsterdam.
9. **Notifications:** your email on failure.
10. Save, then **Run now**.

Compare what you built against `resources/olist_pipeline_job.json`, if you'd rather
paste, the job page has an *Edit as JSON* option (kebab menu, top right). Fix the
notebook paths and the email placeholder if you paste.

**Screenshot:** the DAG view of a completed run, with the `train` task showing as
skipped or run depending on the drift verdict. That image is the whole point.

Note: Free Edition allows a max of 5 concurrent job tasks per account. Your tasks
run sequentially, so you're fine, but don't trigger three runs at once.

---

## Phase 8, Optional: Model Serving (20 min)

Free Edition supports model serving endpoints with limits on how many are active.
If you want serving-both-ways on the CV:

**Models** → `delivery_time_model` → version 1 → **Serve this model** → name the
endpoint → Create. It takes ~10 minutes to provision. Then use the built-in *Query
endpoint* panel to send one of the `input_example` rows.

**Screenshot the endpoint page and a successful query, then delete the endpoint** , 
an idle endpoint eats your quota, and blown quota shuts down your compute for the
rest of the day.

Value: you can then say you deployed the same model two ways, self-managed FastAPI
in a container, and a managed endpoint, and explain when you'd pick each. That
comparison is a genuinely good interview answer.

---

## Phase 9, Get it into the repo (30 min)

1. Commit `databricks/` (four notebooks + `resources/` + `README.md`).
2. Add `databricks/docs/` with your screenshots: workflow DAG, MLflow nested runs,
   UC model with alias, drift table, Catalog tree showing the Delta tables.
3. Update the top-level `README.md`:
   - Replace the architecture diagram with the lakehouse version (the ASCII diagram
     in `databricks/README.md` is a starting point).
   - Add a **Databricks** section pointing at the folder.
   - Update the results table if the MAE shifted, and add one line saying why
     (`randomSplit` ≠ `train_test_split`).
4. Add repo topics on GitHub: `databricks`, `pyspark`, `delta-lake`, `mlflow`,
   `mlops`. Recruiters and GitHub search both use them.

---

## Quota discipline

If you exceed your quota, compute shuts down for the rest of the day, data and
settings survive, but your evening is over. To avoid it:

- Don't leave a serving endpoint running.
- Don't leave the SQL warehouse running after you've finished querying (you get one,
  at `2X-Small`).
- Don't set the job schedule to hourly "to see it run more."
- `RandomForest(n_estimators=200)` is the most expensive thing in the pipeline. If
  you're re-running notebook 03 repeatedly while debugging, drop it to 50 temporarily.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Table or view not found: workspace.olist.orders_clean` | Notebook 01 didn't finish, or the widgets point at a different schema. Check **Catalog**. |
| `%pip install` fails on a network error | Free Edition restricts outbound access to trusted domains. Do the LinkedIn verification, or run the 4-model benchmark. |
| `PERMISSION_DENIED` on `/Shared/olist-delivery-time` | Change the `experiment_path` widget to `/Users/<your-email>/olist-delivery-time`. |
| Model registration fails complaining about a signature | UC requires one. Make sure the `infer_signature` line ran, it's in the same cell. |
| `mlflow.sklearn.log_model() got an unexpected keyword 'name'` | Older MLflow. Change `name="model"` to `artifact_path="model"`. |
| The If/else task can't read the task value | Task key must be exactly `monitor`, and `04_monitor` must have completed successfully. |
| Notebook won't attach to compute | Serverless is warming up; wait ~30s. There's nothing to configure. |
| Quota exhausted mid-session | Nothing to do but wait for the daily reset. See the discipline list above. |

---

## What to be able to answer afterwards

Not filler, these are the questions that actually get asked, and you'll have real
answers only if you did the work rather than skimmed it.

1. Why Delta instead of Parquet? (ACID, schema enforcement, time travel, `DESCRIBE HISTORY`)
2. What is Unity Catalog and what does three-level naming buy you?
3. What broke moving from pandas to PySpark? (all-null `sum()`, `dayofweek` indexing, no `.map()` on columns)
4. Why is your training step *not* distributed?
5. What did Workflows give you over cron? (dependencies, retries, task values, conditional branching)
6. Aliases vs stages in the model registry?
7. When would this pipeline stop working, and what would you change first? (the KS window function is a single-partition sort, it's the first thing to break at scale)
