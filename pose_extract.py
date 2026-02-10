import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

LEG_LMS = [
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
]

def extract_leg_joints(video_path):
    cap = cv2.VideoCapture(video_path)
    joints = []

    MAX_WIDTH = 480
    FRAME_SKIP = 3
    frame_count = 0

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            frame_pts = []
            for idx in LEG_LMS:
                lm = res.pose_landmarks.landmark[idx]
                frame_pts.append([lm.x, lm.y])
            joints.append(frame_pts)

    cap.release()
    return np.array(joints)  # (T, 6, 2)
