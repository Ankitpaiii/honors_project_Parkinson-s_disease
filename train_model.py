import pandas as pd
from custom_bagging import CustomBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv("turning_pd_features.csv")

X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"Training on {len(X)} samples with {X.shape[1]} features...")

# Random Forest style Bagging (Feature Subsampling + Bootstrapping)
# This increases model diversity and usually results in better probability separation.
base_tree = DecisionTreeClassifier(min_samples_leaf=2, random_state=42)
model = CustomBaggingClassifier(
    base_estimator=base_tree, 
    n_estimators=1000, # Increased for smoother probability averages
    max_samples=0.8,   # Use 80% of rows per tree
    max_features=0.8,  # Use 80% of features per tree (Random Forest style)
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved using Custom Bagging Classifier")
