import os
import pandas as pd
from pose_extract import extract_leg_joints
from turning_features import extract_turning_features

DATA = []
LABELS = []
VIDEO_NAMES = []

# Updated to point to the correct folder
ROOT = "Videos"

if not os.path.exists(ROOT):
    print(f"Error: Folder '{ROOT}' not found.")
    exit()

print(f"Scanning folder: {ROOT}")

# Since the folder is flat (no subfolders for classes), we will read all videos
# and assign a placeholder label or try to infer from filename if possible.
# For now, assigning 1 (PD) as most files seem to be PD based on naming (PDFE...).
DEFAULT_LABEL = 1 

try:
    for i, vid in enumerate(os.listdir(ROOT)):
        # if i >= 3: break # LIMIT REMOVED BY USER REQUEST
        if vid.endswith(".mp4"):
            path = os.path.join(ROOT, vid)

            try:
                joints = extract_leg_joints(path)
                if len(joints) < 30:
                    print(f"Skipping {vid}: Not enough data (frames={len(joints)})")
                    continue

                # extract_turning_features returns (features_list, feature_names)
                feats, names = extract_turning_features(joints)

                DATA.append(feats)
                LABELS.append(DEFAULT_LABEL)
                VIDEO_NAMES.append(vid)
                print(f"Processed {vid}")
                
            except Exception as e:
                print(f"Error processing {vid}: {e}")

except KeyboardInterrupt:
    print("\nProcessing interrupted. Saving partial results...")

finally:
    if not DATA:
        print("No features extracted. Exiting.")
        # Only exit if truly empty, otherwise save what we have
        if not os.path.exists("turning_pd_features.csv"):
            exit()
    
    if DATA:
        # SYNTHETIC DATA GENERATION TO ENSURE 2 CLASSES FOR TESTING
        # We duplicate the data and assign label 0 (Control) to the duplicates
        # This is strictly to allow the pipeline to run without valid Control data
        
        current_len = len(DATA)
        print(f"Generating {current_len} synthetic Control samples...")
        for i in range(current_len):
            DATA.append(DATA[i]) # Duplicate features
            LABELS.append(0)     # Assign Control label
            VIDEO_NAMES.append(f"SYNTHETIC_{VIDEO_NAMES[i]}")

        df = pd.DataFrame(DATA, columns=names)
        df["label"] = LABELS
        df["filename"] = VIDEO_NAMES 

        df.to_csv("turning_pd_features.csv", index=False)
        print(f"Feature CSV saved with {len(df)} samples (Half PD, Half Synthetic Control).")


