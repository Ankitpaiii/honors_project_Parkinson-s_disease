import os
# Suppress annoying MediaPipe/TF internal warnings
os.environ['GLOG_minloglevel'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

DATA = []
LABELS = []
VIDEO_NAMES = []

# Training data from actual PD patient videos
ROOT = "Actual videos"

if not os.path.exists(ROOT):
    print(f"Error: Folder '{ROOT}' not found.")
    exit()

print(f"Scanning folder: {ROOT}")

try:
    all_vids = [v for v in sorted(os.listdir(ROOT)) if v.endswith(".mp4")]
    total_vids = len(all_vids)
    print(f"Found {total_vids} video(s) to process.\n")

    for idx, vid in enumerate(all_vids, 1):
        path  = os.path.join(ROOT, vid)
        label = 1  # All "Actual videos" are PD patients
        print(f"  [{idx}/{total_vids}] Processing: {vid} ...", end=" ", flush=True)
        try:
            joints = extract_leg_joints(path)
            if len(joints) < 30:
                print(f"SKIPPED (only {len(joints)} frames)")
                continue
            feats, names = extract_turning_features(joints)
            DATA.append(feats)
            LABELS.append(label)
            VIDEO_NAMES.append(vid)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")

finally:
    if not DATA:
        print("No features extracted.")
        exit()

    print(f"\nExtracted {len(DATA)} real PD samples.")
    print("Building balanced dataset with augmentation & realistic healthy controls...")

    FINAL_DATA = []
    FINAL_LABELS = []
    FINAL_VIDEO_NAMES = []

    np.random.seed(42)

    for i in range(len(DATA)):
        orig = np.array(DATA[i], dtype=float)
        vid = VIDEO_NAMES[i]

        # ── PD AUGMENTATION (10 variations, 5% noise) ──────────────────────
        for j in range(10):
            noise = np.random.normal(1.0, 0.05, len(orig))
            FINAL_DATA.append((orig * noise).tolist())
            FINAL_LABELS.append(1)
            FINAL_VIDEO_NAMES.append(f"PD_{j}_{vid}")

        # ── SYNTHETIC HEALTHY (10 variations, moderate difference + overlap) ─
        # Key insight: healthy people turn faster but NOT perfectly.
        # We keep SOME overlap so the model learns a probabilistic boundary,
        # not a hard rule. This prevents 100% confidence scores.
        #
        # Feature indices:
        #   0: total_turn_angle  → keep similar (both groups turn same angle)
        #   1: mean_angular_velocity → healthy is faster (1.4-1.8x, not 2x)
        #   2: turn_duration     → shorter for healthy (0.5-0.7x, not 0.4x)
        #   3: num_steps         → fewer for healthy (0.5-0.7x)
        #   4: mean_step_length  → slightly larger (1.3-1.6x, not 2x)
        #   5: step_variability  → slightly lower for healthy (0.5-0.8x)
        #   6: freeze_frames     → low but NOT always zero (0-20% of PD value)
        #   7: knee_rom          → slightly larger (1.1-1.2x)

        for j in range(10):
            h = orig.copy()

            # Use per-sample random factors so each healthy twin is unique
            vel_factor      = np.random.uniform(1.3, 1.8)  # faster turn
            dur_factor      = np.random.uniform(0.50, 0.70) # shorter duration
            steps_factor    = np.random.uniform(0.45, 0.65) # fewer steps
            steplen_factor  = np.random.uniform(1.3, 1.6)  # bigger strides
            stepvar_factor  = np.random.uniform(0.45, 0.75) # less shuffling
            freeze_factor   = np.random.uniform(0.0, 0.18) # some freezing (not always 0)
            knee_factor     = np.random.uniform(1.05, 1.20) # slightly more ROM

            h[1] *= vel_factor
            h[2] *= dur_factor
            h[3]  = max(2, int(h[3] * steps_factor))
            h[4] *= steplen_factor
            h[5] *= stepvar_factor
            h[6]  = max(0.0, orig[6] * freeze_factor)  # still has a little freezing
            h[7] *= knee_factor

            # Add 8% noise on top so healthy samples naturally overlap with mild PD
            noise = np.random.normal(1.0, 0.08, len(h))
            FINAL_DATA.append((h * noise).tolist())
            FINAL_LABELS.append(0)
            FINAL_VIDEO_NAMES.append(f"CTRL_{j}_{vid}")

    df = pd.DataFrame(FINAL_DATA, columns=names)
    df["label"] = FINAL_LABELS
    df["filename"] = FINAL_VIDEO_NAMES

    pd_count   = sum(1 for l in FINAL_LABELS if l == 1)
    ctrl_count = sum(1 for l in FINAL_LABELS if l == 0)

    df.to_csv("turning_pd_features.csv", index=False)
    print(f"\nDataset saved: {len(df)} total rows  ({pd_count} PD | {ctrl_count} Control)")
    print("Ready to train. Run: python train_model.py")
