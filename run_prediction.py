import os
# Suppress annoying MediaPipe/TF internal warnings (must be before any imports)
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
import numpy as np
import joblib
import pandas as pd
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

# ── CLI Arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Parkinson's Turning Pattern Analyser")
parser.add_argument("--verbose", action="store_true",
                    help="Print individual feature values per video")
parser.add_argument("--videos", default="Videos",
                    help="Path to folder containing .mp4 files (default: Videos/)")
args = parser.parse_args()

VIDEO_FOLDER = args.videos

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists("model.pkl"):
    print("ERROR: model.pkl not found. Run train_model.py first.")
    sys.exit(1)

if not os.path.exists(VIDEO_FOLDER):
    print(f"ERROR: Video folder '{VIDEO_FOLDER}' not found.")
    sys.exit(1)

model = joblib.load("model.pkl")

print()
print("=" * 60)
print("   PARKINSONIAN TURNING PATTERN ANALYSIS")
print("   Custom Bagging Ensemble | Monte Carlo Inference")
print("=" * 60)
print(f"   Video folder : {VIDEO_FOLDER}")
print(f"   Verbose mode : {'ON' if args.verbose else 'OFF'}")
print("=" * 60)
print()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk_label(prob):
    """3-tier clinical risk label based on probability score."""
    if prob >= 0.65:
        return "HIGH RISK   - Strong Parkinsonian pattern detected"
    elif prob >= 0.35:
        return "BORDERLINE  - Uncertain, clinical review recommended"
    else:
        return "LOW RISK    - Pattern does not strongly suggest PD"

# ── Processing ────────────────────────────────────────────────────────────────
N_TRIALS   = 200
JITTER_STD = 0.06   # 6% jitter mirrors real pose-estimation uncertainty

all_videos = [v for v in sorted(os.listdir(VIDEO_FOLDER)) if v.endswith(".mp4")]
total      = len(all_videos)

if total == 0:
    print(f"No .mp4 files found in '{VIDEO_FOLDER}'.")
    sys.exit(0)

processed, skipped = 0, 0
skipped_list       = []

for i, video in enumerate(all_videos, 1):
    path   = os.path.join(VIDEO_FOLDER, video)
    prefix = f"[{i}/{total}]"

    print(f"{prefix} {video}")

    joints = extract_leg_joints(path)

    if len(joints) < 30:
        reason = f"Only {len(joints)} pose frames detected (need >= 30)"
        print(f"  [WARNING] Skipped — {reason}\n")
        skipped      += 1
        skipped_list.append((video, reason))
        continue

    features, feature_names = extract_turning_features(joints)
    X_base = np.array(features, dtype=float)

    # ── Verbose: show raw feature values ──────────────────────────────────
    if args.verbose:
        print("  Features extracted:")
        for fname, fval in zip(feature_names, X_base):
            print(f"    {fname:<25} : {fval:.4f}")

    # ── Monte Carlo Inference ──────────────────────────────────────────────
    probs = []
    for _ in range(N_TRIALS):
        jitter  = np.random.normal(1.0, JITTER_STD, X_base.shape)
        X_trial = pd.DataFrame([X_base * jitter], columns=feature_names)
        p       = model.predict_proba(X_trial)[0][1]
        probs.append(p)

    prob      = float(np.mean(probs))
    std_score = float(np.std(probs))
    low       = max(0.0, prob - 1.96 * std_score)
    high      = min(1.0, prob + 1.96 * std_score)
    risk      = get_risk_label(prob)

    print(f"  Risk     : {risk}")
    print(f"  Score    : {prob:.3f}  (95% CI: {low:.3f} – {high:.3f})  Uncertainty: {std_score:.3f}")
    print()

    processed += 1

# ── Summary Report ─────────────────────────────────────────────────────────
print("=" * 60)
print("  ANALYSIS SUMMARY")
print("=" * 60)
print(f"  Total videos found   : {total}")
print(f"  Successfully analysed: {processed}")
print(f"  Skipped (errors)     : {skipped}")

if skipped_list:
    print()
    print("  Skipped files:")
    for vid, reason in skipped_list:
        print(f"    - {vid}: {reason}")

print("=" * 60)
