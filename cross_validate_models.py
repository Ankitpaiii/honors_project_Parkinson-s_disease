import pandas as pd
import numpy as np
import sys
import os
import time

# Add root project folder to path for CustomBaggingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from custom_bagging import CustomBaggingClassifier

import warnings
warnings.filterwarnings('ignore')


# ── Load Dataset ──────────────────────────────────────────────────────────────
CSV_PATH = "turning_pd_features.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join("model_all_features", "turning_pd_features.csv")

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print("Dataset not found. Place turning_pd_features.csv in the project root.")
    exit(1)

X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"\nLoaded dataset: {len(X)} samples | {X.shape[1]} features")
print(f"Class distribution: {y.value_counts().to_dict()}")

# ── Model Definitions ─────────────────────────────────────────────────────────
# Note: CustomBaggingClassifier is scaled down (n_estimators=150) for speed
# during 5-fold CV. The production model uses 1000.
_base_tree = DecisionTreeClassifier(min_samples_leaf=4, max_depth=8, random_state=42)

def make_models():
    return {
        "Proposed Bagging (Ours)": CustomBaggingClassifier(
            base_estimator=DecisionTreeClassifier(min_samples_leaf=4, max_depth=8, random_state=42),
            n_estimators=150, max_samples=0.75, max_features=0.75, random_state=42
        ),
        "Random Forest":    RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)":        SVC(kernel="rbf", probability=True),
        "KNN":              KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "ANN (MLP)":        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

# ── Metric Storage ────────────────────────────────────────────────────────────
def init_metrics(model_names):
    return {n: {"acc": [], "precision": [], "recall": [], "f1": [], "error": [], "time": []}
            for n in model_names}

model_names = list(make_models().keys())
metrics_original = init_metrics(model_names)
metrics_pca      = init_metrics(model_names)
metrics_svd      = init_metrics(model_names)

# ── 5-Fold Stratified Cross Validation ───────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nRunning 5-Fold Cross Validation... (this may take a few minutes)\n")

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print(f"  Fold {fold}/5...")

    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

    # -- Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # -- PCA (4 components — same as pca_feature model)
    pca = PCA(n_components=4, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca  = pca.transform(X_test_s)

    # -- SVD (4 components)
    svd = TruncatedSVD(n_components=4, random_state=42)
    X_train_svd = svd.fit_transform(X_train_s)
    X_test_svd  = svd.transform(X_test_s)

    # Recreate fresh models for every fold (avoids state leakage between configs)
    fold_models = make_models()

    for name, model in fold_models.items():
        for tag, X_tr, X_te, store in [
            ("orig", X_train_s,   X_test_s,   metrics_original),
            ("pca",  X_train_pca, X_test_pca, metrics_pca),
            ("svd",  X_train_svd, X_test_svd, metrics_svd),
        ]:
            model_clone = make_models()[name]   # fresh clone per feature set
            model_clone.fit(X_tr, y_train)

            t0    = time.time()
            preds = model_clone.predict(X_te)
            pred_time = time.time() - t0

            store[name]["acc"].append(accuracy_score(y_test, preds))
            store[name]["precision"].append(precision_score(y_test, preds, zero_division=0))
            store[name]["recall"].append(recall_score(y_test, preds, zero_division=0))
            store[name]["f1"].append(f1_score(y_test, preds, zero_division=0))
            store[name]["error"].append(1 - accuracy_score(y_test, preds))
            store[name]["time"].append(pred_time)

print("\nCross validation complete.\n")

# ── Pretty Table Formatter ────────────────────────────────────────────────────
def avg(lst):
    return np.mean(lst)

def print_comparison_table(metrics_orig, metrics_pca, metrics_svd, model_names):
    col_w = 22
    metric_keys = ["acc", "precision", "recall", "f1", "error", "time"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "Error Rate", "Pred Time (s)"]

    for feature_tag, metrics in [
        ("ALL 8 FEATURES (Original)", metrics_orig),
        ("PCA (4 Principal Components)", metrics_pca),
        ("SVD (4 Components)", metrics_svd),
    ]:
        print("\n" + "=" * 120)
        print(f"  RESULTS: {feature_tag}")
        print("=" * 120)

        header = f"{'Model':<{col_w}}"
        for lbl in metric_labels:
            header += f"{lbl:>14}"
        print(header)
        print("-" * 120)

        for name in model_names:
            row = f"{name:<{col_w}}"
            m = metrics[name]
            for key in metric_keys:
                row += f"{avg(m[key]):>14.4f}"
            print(row)
        print("=" * 120)

print_comparison_table(metrics_original, metrics_pca, metrics_svd, model_names)

# ── Grand Summary: Best Model per Metric ──────────────────────────────────────
print("\n" + "=" * 80)
print("  GRAND SUMMARY — BEST MODEL ACROSS ALL FEATURE SETS")
print("=" * 80)

all_results = []
for feature_tag, metrics in [
    ("Original (8 features)", metrics_original),
    ("PCA (4 components)",    metrics_pca),
    ("SVD (4 components)",    metrics_svd),
]:
    for name in model_names:
        m = metrics[name]
        all_results.append({
            "Model": name,
            "Features": feature_tag,
            "Accuracy":  avg(m["acc"]),
            "Precision": avg(m["precision"]),
            "Recall":    avg(m["recall"]),
            "F1 Score":  avg(m["f1"]),
            "Error Rate":avg(m["error"]),
            "Pred Time": avg(m["time"]),
        })

results_df = pd.DataFrame(all_results)

# Best by F1
best_row = results_df.loc[results_df["F1 Score"].idxmax()]
print(f"\n  Best by F1 Score:")
print(f"    Model    : {best_row['Model']}")
print(f"    Features : {best_row['Features']}")
print(f"    Accuracy : {best_row['Accuracy']:.4f}")
print(f"    F1 Score : {best_row['F1 Score']:.4f}")
print(f"    Error    : {best_row['Error Rate']:.4f}")

# Save to CSV
out_path = "model_comparison_results.csv"
results_df.to_csv(out_path, index=False)
print(f"\n  Full results saved to → {out_path}")
print("=" * 80)