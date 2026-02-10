import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("turning_pd_features.csv")

X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved")
