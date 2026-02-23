import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from custom_bagging import CustomBaggingClassifier
import joblib

df = pd.read_csv("turning_pd_features.csv")
X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"Training on {len(X)} samples | {y.value_counts().to_dict()}")

# ── Custom Bagging Ensemble ───────────────────────────────────────────────────
# min_samples_leaf=4 prevents individual trees from perfectly fitting
# the training data — this is the key that creates probability variance
# between trees, giving us non-binary output scores.
base_tree = DecisionTreeClassifier(
    min_samples_leaf=4,     # ← prevents perfect overfitting per tree
    max_depth=8,            # ← caps tree depth so no tree is 100% certain
    random_state=42
)

model = CustomBaggingClassifier(
    base_estimator=base_tree,
    n_estimators=1000,      # 1000 trees for smooth probability averaging
    max_samples=0.75,       # each tree sees 75% of data
    max_features=0.75,      # each tree sees 75% of features
    random_state=42
)

model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model saved: model.pkl")
