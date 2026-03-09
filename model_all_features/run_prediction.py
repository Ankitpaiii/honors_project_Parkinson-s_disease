import os
# Suppress annoying MediaPipe/TF internal warnings
os.environ['GLOG_minloglevel'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
import joblib
import pandas as pd

# Import shared modules from the project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

# Look at the root Videos folder
VIDEO_FOLDER = "../Videos"

if not os.path.exists(VIDEO_FOLDER):
    print(f"Error: {VIDEO_FOLDER} not found.")
    exit(1)

try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully\n")
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")
    exit(1)

print("=" * 55)
print("    PARKINSONIAN TURNING PATTERN ANALYSIS (ALL 8 FEATURES)")
print("=" * 55 + "\n")

for video in sorted(os.listdir(VIDEO_FOLDER)):
    if not video.endswith(".mp4"):
        continue

    path = os.path.join(VIDEO_FOLDER, video)
    joints = extract_leg_joints(path)

    if len(joints) < 30:
        print(f"{video} → Not enough data to analyse\n")
        continue

    features, feature_names = extract_turning_features(joints)
    X_base = np.array(features, dtype=float)

    # ── MONTE CARLO INFERENCE ────────────────────────────────────────────
    N_TRIALS     = 200
    JITTER_STD   = 0.06
    probs = []

    for _ in range(N_TRIALS):
        jitter   = np.random.normal(1.0, JITTER_STD, X_base.shape)
        X_trial  = pd.DataFrame([X_base * jitter], columns=feature_names)
        p        = model.predict_proba(X_trial)[0][1]
        probs.append(p)

    prob = float(np.mean(probs))
    pred = 1 if prob > 0.5 else 0

    std_score  = float(np.std(probs))
    low, high  = prob - 1.96 * std_score, prob + 1.96 * std_score

    if pred == 1:
        result = "MATCHES PARKINSONIAN TURNING PATTERN"
    else:
        result = "LOW SIMILARITY TO PARKINSONIAN PATTERN"

    print(f"{video}")
    print(f"  → {result}")
    print(f"  → Score: {prob:.2f}  (95% CI: {max(0,low):.2f} – {min(1,high):.2f})\n")
