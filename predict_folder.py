import os
import joblib
import pandas as pd
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

VIDEO_FOLDER = "Videos"

model = joblib.load("model.pkl")
print("Model loaded successfully\n")

print("PARKINSONIAN TURNING PATTERN ANALYSIS\n")

for video in os.listdir(VIDEO_FOLDER):
    if video.endswith(".mp4"):
        path = os.path.join(VIDEO_FOLDER, video)

        joints = extract_leg_joints(path)
        if len(joints) < 30:
            print(video, "→ Not enough data")
            continue

        features, feature_names = extract_turning_features(joints)
        X = pd.DataFrame([features], columns=feature_names)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        result = (
            "MATCHES PARKINSONIAN TURNING PATTERN"
            if pred == 1
            else "LOW SIMILARITY TO PARKINSONIAN TURNING PATTERN"
        )

        print(f"{video} → {result} (Score: {prob:.2f})")
