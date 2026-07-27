"""
monitor.py — Olist delivery-time pipeline, Phase 5 (drift monitoring).

Detects data drift by comparing an EARLIER time window (reference: roughly what
the model trained on) against a LATER window (current: newer incoming orders).
For each feature it runs a two-sample Kolmogorov-Smirnov test; a low p-value
means the distribution has shifted. If too many features drift, it flags that a
retrain is warranted -- implementing the thesis's "retraining would be needed"
recommendation.

Run from the repo root:  python src/monitor.py
"""

from pathlib import Path
import json
import pandas as pd
from scipy.stats import ks_2samp

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# Features to monitor (raw columns present in orders_clean.csv), plus the
# target itself -- shifts in delivery_days are the most important to catch.
MONITOR_COLS = [
    "nb_items", "n_sellers", "n_categories",
    "total_price", "total_freight", "total_weight_g",
    "avg_length_cm", "avg_height_cm", "avg_width_cm",
    "purchase_hour", "purchase_dow",
    "delivery_days",
]

P_THRESHOLD = 0.05        # p < 0.05 -> that feature has drifted
DRIFT_SHARE_TRIGGER = 0.5  # if > 50% of features drift -> trigger retrain


# 1. Load and split the data by time
def load_and_split(reference_frac=0.70):
    df = pd.read_csv(PROCESSED / "orders_clean.csv",
                     parse_dates=["order_purchase_timestamp"])
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    cutoff = int(len(df) * reference_frac)
    reference = df.iloc[:cutoff]   # earlier orders (what the model "knows")
    current = df.iloc[cutoff:]     # later orders (new incoming data)
    return reference, current


# 2. Run a KS test per feature
def detect_drift(reference, current):
    results = []
    for col in MONITOR_COLS:
        ref = reference[col].dropna()
        cur = current[col].dropna()
        stat, p_value = ks_2samp(ref, cur)
        results.append({
            "feature": col,
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "drifted": bool(p_value < P_THRESHOLD),
        })
    return results


# 3. Decide whether a retrain is warranted
def summarize(results):
    n_drifted = sum(r["drifted"] for r in results)
    drift_share = n_drifted / len(results)
    trigger = drift_share > DRIFT_SHARE_TRIGGER
    return n_drifted, drift_share, trigger


# 4. Save a simple HTML report (nice screenshot for the README)
def write_html(results, reference, current, n_drifted, drift_share, trigger):
    rows = ""
    for r in results:
        color = "#ffdddd" if r["drifted"] else "#ddffdd"
        flag = "DRIFT" if r["drifted"] else "ok"
        rows += (f"<tr style='background:{color}'><td>{r['feature']}</td>"
                 f"<td>{r['ks_statistic']}</td><td>{r['p_value']}</td>"
                 f"<td>{flag}</td></tr>")

    verdict = ("RETRAIN TRIGGERED" if trigger else "No retrain needed")
    ref_range = (f"{reference['order_purchase_timestamp'].min().date()} to "
                 f"{reference['order_purchase_timestamp'].max().date()}")
    cur_range = (f"{current['order_purchase_timestamp'].min().date()} to "
                 f"{current['order_purchase_timestamp'].max().date()}")

    html = f"""<html><head><title>Drift Report</title>
<style>body{{font-family:sans-serif;margin:40px}} table{{border-collapse:collapse}}
td,th{{border:1px solid #999;padding:6px 12px;text-align:left}}</style></head>
<body>
<h1>Data Drift Report</h1>
<p><b>Reference window:</b> {ref_range} ({len(reference):,} orders)<br>
<b>Current window:</b> {cur_range} ({len(current):,} orders)</p>
<p><b>Features drifted:</b> {n_drifted}/{len(results)} ({drift_share:.0%})<br>
<b>Verdict:</b> {verdict}</p>
<table><tr><th>Feature</th><th>KS statistic</th><th>p-value</th><th>Status</th></tr>
{rows}</table></body></html>"""

    (REPORTS / "drift_report.html").write_text(html, encoding="utf-8")


# 5. Run everything
def main():
    reference, current = load_and_split()
    results = detect_drift(reference, current)
    n_drifted, drift_share, trigger = summarize(results)

    print(f"Reference: {len(reference):,} orders  |  Current: {len(current):,} orders\n")
    print(f"{'feature':<26}{'KS stat':>10}{'p-value':>12}   status")
    for r in results:
        status = "DRIFT" if r["drifted"] else "ok"
        print(f"{r['feature']:<26}{r['ks_statistic']:>10}{r['p_value']:>12}   {status}")

    print(f"\nFeatures drifted: {n_drifted}/{len(results)} ({drift_share:.0%})")
    if trigger:
        print(">>> RETRAIN TRIGGERED: drift exceeds threshold, retrain the model.")
    else:
        print(">>> No retrain needed: drift within acceptable bounds.")

    # Save machine-readable + human-readable reports.
    (REPORTS / "drift_report.json").write_text(
        json.dumps({"n_drifted": n_drifted, "drift_share": drift_share,
                    "retrain_triggered": trigger, "features": results}, indent=2),
        encoding="utf-8",
    )
    write_html(results, reference, current, n_drifted, drift_share, trigger)
    print(f"\nSaved -> {REPORTS/'drift_report.json'} and {REPORTS/'drift_report.html'}")


if __name__ == "__main__":
    main()