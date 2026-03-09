import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from custom_bagging import CustomBaggingClassifier
import joblib

df = pd.read_csv("turning_pd_features.csv")
X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"Loaded dataset: {len(X)} samples | Class distribution: {y.value_counts().to_dict()}")

# ── Hold-out split for validation reporting ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} samples | Test (hold-out): {len(X_test)} samples\n")

# ── Custom Bagging Ensemble ───────────────────────────────────────────────────
# min_samples_leaf=4 prevents individual trees from perfectly fitting the
# training data — this creates probability variance between trees, giving
# us non-binary output scores (e.g. 0.87 instead of 1.0).
base_tree = DecisionTreeClassifier(
    min_samples_leaf=4,   # prevents perfect per-tree overfiting
    max_depth=8,          # caps tree depth so no tree is 100% certain
    random_state=42
)

model = CustomBaggingClassifier(
    base_estimator=base_tree,
    n_estimators=1000,    # 1000 trees for smooth probability averaging
    max_samples=0.75,     # each tree sees 75% of data
    max_features=0.75,    # each tree sees 75% of features
    random_state=42
)

print("Training model on full dataset...")
model.fit(X, y)

# ── Validation Report ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING VALIDATION REPORT")
print("=" * 55)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"  Hold-out Test Accuracy : {acc * 100:.2f}%")
print(f"  Hold-out Test Samples  : {len(y_test)}")
print()
print(classification_report(y_test, y_pred, target_names=["Healthy Control", "Parkinson's"]))
print("=" * 55)

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model, "model.pkl")
print(f"\nModel saved: model.pkl")
print("Train complete. Run: python run_prediction.py")
