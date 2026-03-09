import os
# Suppress MediaPipe C++ logs (must be set before importing mediapipe)
os.environ['GLOG_minloglevel'] = '3' 

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import os

# ── Landmark indices (same meaning as the old mp.solutions.pose) ─────────────
# 23: LEFT_HIP    24: RIGHT_HIP
# 25: LEFT_KNEE   26: RIGHT_KNEE
# 27: LEFT_ANKLE  28: RIGHT_ANKLE
LEG_LM_INDICES = [23, 24, 25, 26, 27, 28]

# ── Load the pose landmarker model ──────────────────────────────────────────
# Looks for the task file in the same folder as this script, then one level up
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "pose_landmarker.task")
if not os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "pose_landmarker.task")
if not os.path.exists(_MODEL_PATH):
    raise FileNotFoundError(
        "pose_landmarker.task not found. "
        "Please ensure it exists in the project folder."
    )

_base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
_options = vision.PoseLandmarkerOptions(
    base_options=_base_options,
    output_segmentation_masks=False,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_leg_joints(video_path):
    cap = cv2.VideoCapture(video_path)
    joints = []

    MAX_WIDTH  = 480
    FRAME_SKIP = 3
    frame_count = 0

    with vision.PoseLandmarker.create_from_options(_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize for speed
            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                scale = MAX_WIDTH / w
                frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)))

            # Convert BGR → RGB for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                frame_pts = []
                for idx in LEG_LM_INDICES:
                    lm = landmarks[idx]
                    frame_pts.append([lm.x, lm.y])
                joints.append(frame_pts)

    cap.release()
    return np.array(joints)  # shape: (T, 6, 2)
