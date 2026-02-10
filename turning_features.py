import numpy as np

def extract_turning_features(joints, fps=30):
    feats = {}

    L_hip, R_hip = joints[:,0], joints[:,1]
    L_ankle, R_ankle = joints[:,4], joints[:,5]

    # 1️⃣ Hip orientation (turning angle)
    hip_vec = R_hip - L_hip
    angles = np.unwrap(np.arctan2(hip_vec[:,1], hip_vec[:,0]))

    feats["total_turn_angle"] = abs(angles[-1] - angles[0])
    feats["mean_angular_velocity"] = np.mean(np.abs(np.diff(angles) * fps))
    feats["turn_duration"] = len(joints) / fps

    # 2️⃣ Number of steps during turning
    ankle_speed = np.linalg.norm(np.diff((L_ankle + R_ankle)/2, axis=0), axis=1)
    feats["num_steps"] = np.sum(ankle_speed > 0.015)

    # 3️⃣ Step length variability (shuffling indicator)
    step_L = np.linalg.norm(np.diff(L_ankle, axis=0), axis=1)
    step_R = np.linalg.norm(np.diff(R_ankle, axis=0), axis=1)

    feats["mean_step_length"] = np.mean((step_L + step_R) / 2)
    feats["step_variability"] = np.std((step_L + step_R) / 2)

    # 4️⃣ Freezing of gait proxy
    feats["freeze_frames"] = np.sum(ankle_speed < 0.01)

    # 5️⃣ Knee range of motion
    def knee_angle(h, k, a):
        v1, v2 = h-k, a-k
        return np.arccos(
            np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)
        )

    L_angles, R_angles = [], []
    for i in range(len(joints)):
        L_angles.append(knee_angle(joints[i,0], joints[i,2], joints[i,4]))
        R_angles.append(knee_angle(joints[i,1], joints[i,3], joints[i,5]))

    feats["knee_rom"] = (np.max(L_angles)+np.max(R_angles))/2 - \
                        (np.min(L_angles)+np.min(R_angles))/2

    return list(feats.values()), list(feats.keys())
