"""
features.py — Olist delivery-time pipeline, Phase 1 (feature engineering).

Takes data/processed/orders_clean.csv and produces model-ready train/test sets.
Every transform that "learns" from data (medians for imputation, median
encodings) is fit on the training split, then applied to the test split,
so no information leaks from test into training.

Run from the repo root:   python src/features.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed")

# Column roles
TARGET = "delivery_days"

# Numeric features fed straight to the model
NUMERIC = [
    "nb_items", "n_sellers", "n_categories",
    "total_price", "total_freight", "total_weight_g",
    "avg_length_cm", "avg_height_cm", "avg_width_cm",
    "purchase_hour", "purchase_dow",
]

# Numeric columns that have missing values to impute (the 16 rows)
IMPUTE = ["avg_length_cm", "avg_height_cm", "avg_width_cm"]

# High-cardinality categoricals -> median target encoding
MEDIAN_ENCODE = ["customer_city", "customer_zip_code_prefix"]

# Low-cardinality categorical -> one-hot encoding.
ONE_HOT = ["customer_state"]

# Columns we keep around but don't feed to the model (needed later or IDs).
DROP_FROM_X = ["order_id", "order_purchase_timestamp", "purchase_month"]


#1. Load
def load_clean():
    df = pd.read_csv(PROCESSED / "orders_clean.csv")
    return df


#2. Split  
def split(df):
    """Random 80/20 split with a fixed seed"""
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    return train_df.copy(), test_df.copy()


#3. Impute missing
def impute(train_df, test_df):
    for col in IMPUTE:
        median = train_df[col].median()          
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)
    return train_df, test_df


#4. Median-encode high-cardinality categoricals
def median_encode(train_df, test_df):
    """
    For each high-card column, replace the category with the median target
    value of that category, computed on TRAIN. Unseen categories in test fall
    back to the global train median. Produces a new *_med column.
    """
    global_median = train_df[TARGET].median()
    for col in MEDIAN_ENCODE:
        mapping = train_df.groupby(col)[TARGET].median()     
        new_col = f"{col}_med"
        train_df[new_col] = train_df[col].map(mapping)
        test_df[new_col] = test_df[col].map(mapping).fillna(global_median)
    return train_df, test_df


#5. One-hot encode low-cardinality categoricals
def one_hot(train_df, test_df):
    train_oh = pd.get_dummies(train_df[ONE_HOT], prefix=ONE_HOT)
    test_oh = pd.get_dummies(test_df[ONE_HOT], prefix=ONE_HOT)
    # Align test columns to train (test may be missing a rare category)
    test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0)
    return train_oh, test_oh


#6. Assemble final matrices
def build_features():
    df = load_clean()
    train_df, test_df = split(df)

    train_df, test_df = impute(train_df, test_df)
    train_df, test_df = median_encode(train_df, test_df)
    train_oh, test_oh = one_hot(train_df, test_df)

    med_cols = [f"{c}_med" for c in MEDIAN_ENCODE]
    feature_cols = NUMERIC + med_cols

    # Numeric and median-encoded features, then one-hot columns
    X_train = pd.concat(
        [train_df[feature_cols].reset_index(drop=True), train_oh.reset_index(drop=True)],
        axis=1,
    )
    X_test = pd.concat(
        [test_df[feature_cols].reset_index(drop=True), test_oh.reset_index(drop=True)],
        axis=1,
    )
    y_train = train_df[TARGET].reset_index(drop=True)
    y_test = test_df[TARGET].reset_index(drop=True)

    # Save target alongside features so train.py can just read these
    train_out = X_train.copy()
    train_out[TARGET] = y_train
    test_out = X_test.copy()
    test_out[TARGET] = y_test

    train_out.to_csv(PROCESSED / "train.csv", index=False)
    test_out.to_csv(PROCESSED / "test.csv", index=False)

    return X_train, X_test, y_train, y_test


#7. Run + sanity report
def main():
    X_train, X_test, y_train, y_test = build_features()

    print(f"Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} rows, {X_test.shape[1]} features\n")
    print(f"Feature columns: {list(X_train.columns)}\n")
    print(f"Any missing in X_train? {X_train.isna().any().any()}")
    print(f"Any missing in X_test?  {X_test.isna().any().any()}")
    print(f"\nSaved -> {PROCESSED / 'train.csv'} and {PROCESSED / 'test.csv'}")


if __name__ == "__main__":
    main()
