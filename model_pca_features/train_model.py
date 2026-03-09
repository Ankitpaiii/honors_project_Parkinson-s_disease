import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Reference shared files from the project root
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from custom_bagging import CustomBaggingClassifier

# Use the root dataset
df = pd.read_csv(os.path.join(ROOT, "turning_pd_features.csv"))
X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"Training on {len(X)} samples | {y.value_counts().to_dict()}")

# ── PCA Preprocessing Pipeline ───────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=4)   # Retain top 4 principal components
X_pca = pca.fit_transform(X_scaled)

print(f"PCA Reduced shape: {X_pca.shape}")

# ── Custom Bagging Ensemble ───────────────────────────────────────────────────
base_tree = DecisionTreeClassifier(
    min_samples_leaf=4,
    max_depth=8,
    random_state=42
)

model = CustomBaggingClassifier(
    base_estimator=base_tree,
    n_estimators=1000,
    max_samples=0.75,
    max_features=0.75,
    random_state=42
)

model.fit(X_pca, y)
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca,    "pca.pkl")
print("Model, Scaler, and PCA saved: model.pkl | scaler.pkl | pca.pkl (Using 4 Principal Components)")
