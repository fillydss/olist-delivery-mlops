"""
ingest.py — Olist delivery-time pipeline, Phase 1 data prep.

Loads the raw Olist CSVs, computes the delivery-time target, filters to
completed (delivered) orders, aggregates order-level features to ONE row per
order, joins customer geography, and writes a clean table to data/processed/.

Run from the repo root:   python src/ingest.py
"""

from pathlib import Path
import pandas as pd

# --- Paths -------------------------------------------------------------------
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


# --- 1. Load -----------------------------------------------------------------
def load_raw():
    """Read the raw CSVs we need. Timestamp columns are parsed as datetimes."""
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    return orders, items, products, customers


# --- 2. Target + filter ------------------------------------------------------
def build_target_and_filter(orders):
    """
    Keep only genuine, completed delivery events, then compute the target:
    delivery time in DAYS = delivered_customer_date - purchase_timestamp.
    """
    # Keep only delivered orders with a real delivery timestamp.
    df = orders[orders["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"])

    # Target in days (change to hours with .../ 3600 on total_seconds if you prefer).
    delta = df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    df["delivery_days"] = delta.dt.total_seconds() / 86400

    # Drop impossible values (data-entry errors: negative or zero durations).
    df = df[df["delivery_days"] > 0]

    return df


# --- 3. Aggregate items to one row per order ---------------------------------
def aggregate_items(items, products):
    """
    order_items has MULTIPLE rows per order. Join product attributes, then
    collapse to one row per order with count/sum/mean aggregates.
    """
    items = items.merge(
        products[
            ["product_id", "product_category_name", "product_weight_g",
             "product_length_cm", "product_height_cm", "product_width_cm"]
        ],
        on="product_id",
        how="left",
    )

    agg = items.groupby("order_id").agg(
        nb_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("product_category_name", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        avg_length_cm=("product_length_cm", "mean"),
        avg_height_cm=("product_height_cm", "mean"),
        avg_width_cm=("product_width_cm", "mean"),
    ).reset_index()

    return agg


# --- 4. Join everything into the final order-level table ----------------------
def build_dataset():
    orders, items, products, customers = load_raw()

    df = build_target_and_filter(orders)
    item_agg = aggregate_items(items, products)

    # Customer geography (high-cardinality categoricals for later encoding).
    cust = customers[["customer_id", "customer_state", "customer_zip_code_prefix", "customer_city"]]

    df = (
        df.merge(item_agg, on="order_id", how="inner")   # inner: keep orders that have items
          .merge(cust, on="customer_id", how="left")
    )

    # A couple of cheap time features from the purchase timestamp.
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    # Keep the columns the model will use, plus the timestamp (needed for the
    # time-based split / drift simulation in Phase 5) and the target.
    keep = [
        "order_id", "order_purchase_timestamp", "delivery_days",
        "nb_items", "n_sellers", "n_categories",
        "total_price", "total_freight", "total_weight_g",
        "avg_length_cm", "avg_height_cm", "avg_width_cm",
        "customer_state", "customer_zip_code_prefix", "customer_city",
        "purchase_hour", "purchase_dow", "purchase_month",
    ]
    return df[keep]


# --- 5. Run + sanity report --------------------------------------------------
def main():
    df = build_dataset()

    print(f"Final rows (orders): {len(df):,}")
    print(f"Columns: {list(df.columns)}\n")
    print("Target (delivery_days) summary:")
    print(df["delivery_days"].describe(), "\n")
    print("Missing values per column:")
    print(df.isna().sum())

    out = PROCESSED / "orders_clean.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
