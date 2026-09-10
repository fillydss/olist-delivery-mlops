# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01 Ingest
# MAGIC
# MAGIC PySpark port of `src/ingest.py`. Reads the raw Olist CSVs from a Unity Catalog
# MAGIC Volume, computes the delivery-time target, aggregates order items to one row per
# MAGIC order, joins customer geography, and writes a **Delta** table.
# MAGIC
# MAGIC Output: `{catalog}.{schema}.orders_clean`

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "olist")
dbutils.widgets.text("volume", "raw")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
TARGET = "delivery_days"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Reading from {RAW}")

# COMMAND ----------

from pyspark.sql import functions as F

# --- 1. Load ------------------------------------------------------------------
# inferSchema handles the timestamp columns
def read_csv(name):
    return (spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(f"{RAW}/{name}"))

orders = read_csv("olist_orders_dataset.csv")
items = read_csv("olist_order_items_dataset.csv")
products = read_csv("olist_products_dataset.csv")
customers = read_csv("olist_customers_dataset.csv")

print(f"raw orders: {orders.count():,}   items: {items.count():,}")

# COMMAND ----------

# --- 2. Target + filter -------------------------------------------------------
# delivery_days = delivered_customer_date - purchase_timestamp, in days.
orders_delivered = (
    orders
    .filter(F.col("order_status") == "delivered")
    .filter(F.col("order_delivered_customer_date").isNotNull()
            & F.col("order_purchase_timestamp").isNotNull())
    .withColumn(
        TARGET,
        (F.col("order_delivered_customer_date").cast("double")
         - F.col("order_purchase_timestamp").cast("double")) / 86400.0,
    )
    .filter(F.col(TARGET) > 0)  # drop impossible negative/zero durations
)

# COMMAND ----------

# --- 3. Aggregate items to one row per order ----------------------------------
items_enriched = items.join(
    products.select(
        "product_id", "product_category_name", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ),
    on="product_id",
    how="left",
)

item_agg = items_enriched.groupBy("order_id").agg(
    F.count("order_item_id").alias("nb_items"),
    F.countDistinct("seller_id").alias("n_sellers"),
    F.countDistinct("product_category_name").alias("n_categories"),
    F.sum("price").alias("total_price"),
    F.sum("freight_value").alias("total_freight"),
    F.coalesce(F.sum("product_weight_g"), F.lit(0.0)).alias("total_weight_g"),
    F.avg("product_length_cm").alias("avg_length_cm"),
    F.avg("product_height_cm").alias("avg_height_cm"),
    F.avg("product_width_cm").alias("avg_width_cm"),
)

# COMMAND ----------

# --- 4. Join into the final order-level table ---------------------------------
cust = customers.select(
    "customer_id", "customer_state", "customer_zip_code_prefix", "customer_city"
)

# broadcast: customers/items aggregates are small relative to orders, so this
# avoids a full shuffle join. Spark's AQE would often pick this anyway; being
# explicit documents the intent.
orders_clean = (
    orders_delivered
    .join(F.broadcast(item_agg), on="order_id", how="inner")
    .join(F.broadcast(cust), on="customer_id", how="left")
    # dayofweek() is Sunday=1..Saturday=7; pandas' dayofweek is Monday=0..Sunday=6
    .withColumn("purchase_hour", F.hour("order_purchase_timestamp"))
    .withColumn("purchase_dow", (F.dayofweek("order_purchase_timestamp") + 5) % 7)
    .withColumn("purchase_month", F.date_format("order_purchase_timestamp", "yyyy-MM"))
    .select(
        "order_id", "order_purchase_timestamp", TARGET,
        "nb_items", "n_sellers", "n_categories",
        "total_price", "total_freight", "total_weight_g",
        "avg_length_cm", "avg_height_cm", "avg_width_cm",
        "customer_state", "customer_zip_code_prefix", "customer_city",
        "purchase_hour", "purchase_dow", "purchase_month",
    )
)

# COMMAND ----------

# --- 5. Write Delta -----------------------------------------------------------
(orders_clean.write
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.orders_clean"))

n = spark.table(f"{CATALOG}.{SCHEMA}.orders_clean").count()
print(f"Wrote {n:,} rows -> {CATALOG}.{SCHEMA}.orders_clean")

# COMMAND ----------

# --- 6. Sanity report ---------------------------------------------------------
df = spark.table(f"{CATALOG}.{SCHEMA}.orders_clean")
display(df.select(TARGET).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max"))

# COMMAND ----------

# Nulls per column
display(df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
]))

# COMMAND ----------

# MAGIC %md
# MAGIC Delta gives us table history for free, useful to see what
# MAGIC the model trained on.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.{SCHEMA}.orders_clean"))

# COMMAND ----------

dbutils.jobs.taskValues.set("orders_clean_rows", n)