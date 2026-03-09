import pandas as pd
import numpy as np

# ── Load saved results ────────────────────────────────────────────────────────
try:
    df = pd.read_csv("model_comparison_results.csv")
except FileNotFoundError:
    print("ERROR: model_comparison_results.csv not found.")
    print("Please run compare_models.py first to generate results.")
    exit(1)

# ── Table printing helper ─────────────────────────────────────────────────────
COL_HEADERS = [
    "Classifier Models",
    "Precision",
    "Recall",
    "F1-Score",
    "Prediction Time",
    "Prediction Accuracy",
    "Prediction Error Rate",
]

CSV_COLS = ["Model", "Precision", "Recall", "F1 Score", "Pred Time", "Accuracy", "Error Rate"]

# Column widths
W_MODEL   = 30
W_METRIC  = 21

TOTAL_W = W_MODEL + W_METRIC * (len(COL_HEADERS) - 1)

def print_table(title, feature_tag):
    subset = df[df["Features"] == feature_tag].copy()

    print("\n")
    print("=" * TOTAL_W)
    print(f"  {title}")
    print("=" * TOTAL_W)

    # Header row
    header = f"{'Classifier Models':<{W_MODEL}}"
    for col in COL_HEADERS[1:]:
        header += f"{col:>{W_METRIC}}"
    print(header)
    print("-" * TOTAL_W)

    # Data rows
    for _, row in subset.iterrows():
        line = f"{row['Model']:<{W_MODEL}}"
        line += f"{float(row['Precision']):>{W_METRIC}.4f}"
        line += f"{float(row['Recall']):>{W_METRIC}.4f}"
        line += f"{float(row['F1 Score']):>{W_METRIC}.4f}"
        line += f"{float(row['Pred Time']):>{W_METRIC}.6f}"
        line += f"{float(row['Accuracy']):>{W_METRIC}.4f}"
        line += f"{float(row['Error Rate']):>{W_METRIC}.4f}"
        print(line)

    print("=" * TOTAL_W)

    # Quick best-model summary for this table
    best_idx = subset["F1 Score"].astype(float).idxmax()
    best = subset.loc[best_idx]
    print(f"  >> Best F1: {best['Model']}  ->  F1={float(best['F1 Score']):.4f}  |  Accuracy={float(best['Accuracy']):.4f}")
    print("=" * TOTAL_W)


# ── Print all 3 tables ────────────────────────────────────────────────────────
print_table(
    title       = "Comparative Analysis Without Feature Selection",
    feature_tag = "Original (8 features)"
)

print_table(
    title       = "Comparative Analysis With PCA Feature Selection",
    feature_tag = "PCA (4 components)"
)

print_table(
    title       = "Comparative Analysis With SVD Feature Selection",
    feature_tag = "SVD (4 components)"
)

# ── Overall best across all 3 tables ─────────────────────────────────────────
print("\n")
print("=" * TOTAL_W)
print("  OVERALL BEST MODEL — ACROSS ALL FEATURE SETS")
print("=" * TOTAL_W)

best_overall_idx = df["F1 Score"].astype(float).idxmax()
best_overall = df.loc[best_overall_idx]

print(f"  Classifier Models   : {best_overall['Model']}")
print(f"  Feature Set         : {best_overall['Features']}")
print(f"  Prediction Accuracy : {float(best_overall['Accuracy']):.4f}")
print(f"  Precision           : {float(best_overall['Precision']):.4f}")
print(f"  Recall              : {float(best_overall['Recall']):.4f}")
print(f"  F1-Score            : {float(best_overall['F1 Score']):.4f}")
print(f"  Prediction Error Rate:{float(best_overall['Error Rate']):.4f}")
print(f"  Prediction Time     : {float(best_overall['Pred Time']):.6f} sec")
print("=" * TOTAL_W)
