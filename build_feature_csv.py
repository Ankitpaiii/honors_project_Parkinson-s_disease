import os
import pandas as pd
import numpy as np
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

DATA = []
LABELS = []
VIDEO_NAMES = []

# Updated to point to the correct folder for training data
ROOT = "Actual videos"

if not os.path.exists(ROOT):
    print(f"Error: Folder '{ROOT}' not found.")
    exit()

print(f"Scanning folder: {ROOT}")

try:
    for vid in os.listdir(ROOT):
        if vid.endswith(".mp4"):
            path = os.path.join(ROOT, vid)

            # Assign label 1 (PD) to all videos in "Actual videos"
            label = 1
            
            try:
                joints = extract_leg_joints(path)
                if len(joints) < 30: 
                    print(f"Skipping {vid}: Not enough data")
                    continue

                feats, names = extract_turning_features(joints)
                
                DATA.append(feats)
                LABELS.append(label)
                VIDEO_NAMES.append(vid)
                
                print(f"Processed {vid}")
                
            except Exception as e:
                print(f"Error processing {vid}: {e}")

except KeyboardInterrupt:
    print("\nProcessing interrupted.")

finally:
    if not DATA:
        print("No features extracted.")
        exit()
    
    if DATA:
        # Since "Actual videos" only has PD data, generate synthetic healthy samples
        print(f"Expanding dataset with 5x augmentation and synthetic controls...")
        
        FINAL_DATA = []
        FINAL_LABELS = []
        FINAL_VIDEO_NAMES = []

        for i in range(len(DATA)):
            orig_feats = DATA[i]
            orig_vid = VIDEO_NAMES[i]
            
            # --- 1. PD AUGMENTATION (5 variations of the real patient) ---
            for j in range(5):
                # Add 3% random jitter
                noise = np.random.normal(1, 0.03, len(orig_feats))
                aug_pd = (np.array(orig_feats) * noise).tolist()
                FINAL_DATA.append(aug_pd)
                FINAL_LABELS.append(1)
                FINAL_VIDEO_NAMES.append(f"AUG_{j}_{orig_vid}")

            # --- 2. SYNTHETIC HEALTHY GENERATION (5 variations of a 'healthy' twin) ---
            # Create the 'prototype' healthy version
            healthy_proto = list(orig_feats)
            healthy_proto[1] *= 2.0  # Double the angular velocity (faster turning)
            healthy_proto[2] /= 2.5  # Much shorter duration
            healthy_proto[3] = max(2, int(healthy_proto[3] * 0.3)) # Fewer steps
            healthy_proto[4] *= 2.0  # Larger step length
            healthy_proto[5] *= 0.4  # Less variability
            healthy_proto[6] = 0     # Zero freezing
            healthy_proto[7] *= 1.3  # More Knee ROM
            
            for j in range(5):
                noise = np.random.normal(1, 0.03, len(healthy_proto))
                aug_healthy = (np.array(healthy_proto) * noise).tolist()
                FINAL_DATA.append(aug_healthy)
                FINAL_LABELS.append(0)
                FINAL_VIDEO_NAMES.append(f"SYNTH_AUG_{j}_{orig_vid}")

        df = pd.DataFrame(FINAL_DATA, columns=names)
        df["label"] = FINAL_LABELS
        df["filename"] = FINAL_VIDEO_NAMES 

        df.to_csv("turning_pd_features.csv", index=False)
        print(f"Feature CSV saved with {len(df)} samples (Balanced & Augmented).")
