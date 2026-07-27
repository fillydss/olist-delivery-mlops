"""
features.py — Olist delivery-time pipeline, Phase 1 (feature engineering).

Takes data/processed/orders_clean.csv and produces model-ready train/test sets.
Every transform that "learns" from data (medians for imputation, median
encodings) is fit ONLY on the training split, then applied to test, so no
information leaks. It also SAVES the fitted encoders + column order to
artifacts/ so the prediction API can reproduce the exact same transformation.

Run from the repo root:   python src/features.py
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TARGET = "delivery_days"

NUMERIC = [
    "nb_items", "n_sellers", "n_categories",
    "total_price", "total_freight", "total_weight_g",
    "avg_length_cm", "avg_height_cm", "avg_width_cm",
    "purchase_hour", "purchase_dow",
]
IMPUTE = ["avg_length_cm", "avg_height_cm", "avg_width_cm"]
MEDIAN_ENCODE = ["customer_city", "customer_zip_code_prefix"]
ONE_HOT = ["customer_state"]


# 1. Load
def load_clean():
    return pd.read_csv(PROCESSED / "orders_clean.csv")


# 2. Split first (before fitting anything)
def split(df):
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    return train_df.copy(), test_df.copy()


# 3. Impute missing numeric values with TRAIN medians (and remember them)
def impute(train_df, test_df):
    impute_values = {}
    for col in IMPUTE:
        median = train_df[col].median()
        impute_values[col] = median
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)
    return train_df, test_df, impute_values


# 4. Median-encode high-cardinality categoricals (and remember the maps)
def median_encode(train_df, test_df):
    global_median = train_df[TARGET].median()
    maps = {}
    for col in MEDIAN_ENCODE:
        mapping = train_df.groupby(col)[TARGET].median()
        maps[col] = mapping.to_dict()
        new_col = f"{col}_med"
        train_df[new_col] = train_df[col].map(mapping)
        test_df[new_col] = test_df[col].map(mapping).fillna(global_median)
    return train_df, test_df, maps, global_median


# 5. One-hot encode low-cardinality categoricals
def one_hot(train_df, test_df):
    train_oh = pd.get_dummies(train_df[ONE_HOT], prefix=ONE_HOT)
    test_oh = pd.get_dummies(test_df[ONE_HOT], prefix=ONE_HOT)
    test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0)
    return train_oh, test_oh


# 6. Assemble matrices and save everything the API will need
def build_features():
    df = load_clean()
    train_df, test_df = split(df)

    train_df, test_df, impute_values = impute(train_df, test_df)
    train_df, test_df, median_maps, global_median = median_encode(train_df, test_df)
    train_oh, test_oh = one_hot(train_df, test_df)

    med_cols = [f"{c}_med" for c in MEDIAN_ENCODE]
    feature_cols = NUMERIC + med_cols

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

    train_out = X_train.copy(); train_out[TARGET] = y_train
    test_out = X_test.copy(); test_out[TARGET] = y_test
    train_out.to_csv(PROCESSED / "train.csv", index=False)
    test_out.to_csv(PROCESSED / "test.csv", index=False)

    # Save the fitted transformation so the API can reproduce it exactly.
    preprocessor = {
        "impute_values": impute_values,
        "median_maps": median_maps,
        "global_median": global_median,
        "feature_columns": list(X_train.columns),   # exact order the model expects
    }
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")

    return X_train, X_test, y_train, y_test


# 7. Run + sanity report
def main():
    X_train, X_test, y_train, y_test = build_features()
    print(f"Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} rows, {X_test.shape[1]} features")
    print(f"Any missing in X_train? {X_train.isna().any().any()}")
    print(f"Any missing in X_test?  {X_test.isna().any().any()}")
    print(f"Saved -> {PROCESSED/'train.csv'}, {PROCESSED/'test.csv'}, "
          f"{ARTIFACTS/'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
