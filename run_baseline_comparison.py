import os
# Suppress annoying MediaPipe/TF internal warnings
os.environ['GLOG_minloglevel'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pose_extract import extract_leg_joints
from turning_features import extract_turning_features
from sklearn.preprocessing import StandardScaler

print("Loading dataset to quickly train models...")
df = pd.read_csv("turning_pd_features.csv")
X_train_full = df.drop(["label", "filename"], axis=1, errors="ignore").values
y_train_full = df["label"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_full)

# ==========================================================
# 1. Define and train Models
# ==========================================================
models_dict = {}

try:
    from custom_bagging import CustomBaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    base_tree = DecisionTreeClassifier(min_samples_leaf=4, max_depth=8, random_state=42)
    proposed = CustomBaggingClassifier(
        base_estimator=base_tree, n_estimators=1000, 
        max_samples=0.75, max_features=0.75, random_state=42
    )
    proposed.fit(X_train_full, y_train_full)
    models_dict["Proposed Bagging"] = (proposed, False) # False means not Keras
except Exception as e:
    print("Could not train proposed model:", e)

from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
ann.fit(X_train_s, y_train_full)
models_dict["RNN / ANN"] = (ann, False)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
tf.random.set_seed(42)

X_train_cnn = np.expand_dims(X_train_s, axis=2)

cnn = Sequential([
    Conv1D(16, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_cnn, y_train_full, epochs=50, batch_size=16, verbose=0)
models_dict["Simple CNN"] = (cnn, True)

def build_custom_cnn(depth=3, dense=False):
    inputs = tf.keras.Input(shape=(8, 1))
    x = inputs
    for _ in range(depth):
        conv = Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
        if dense: x = tf.keras.layers.Concatenate()([x, conv])
        else: x = conv
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

eff = build_custom_cnn(depth=3, dense=False)
eff.fit(X_train_cnn, y_train_full, epochs=20, batch_size=16, verbose=0)
models_dict["Deep CNN (3-Layer)"] = (eff, True)

dense_net = build_custom_cnn(depth=4, dense=True)
dense_net.fit(X_train_cnn, y_train_full, epochs=20, batch_size=16, verbose=0)
models_dict["Dense CNN (4-Layer)"] = (dense_net, True)

vgg = Sequential([
    Conv1D(8, 3, padding='same', activation='relu', input_shape=(8, 1)),
    Conv1D(8, 3, padding='same', activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(1, activation='sigmoid')
])
vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
vgg.fit(X_train_cnn, y_train_full, epochs=20, batch_size=16, verbose=0)
models_dict["Shallow CNN (2-Layer)"] = (vgg, True)

print("Training complete.\n")

# ==========================================================
# 2. Prediction on Videos Folder
# ==========================================================
VIDEO_FOLDER = "Videos"
print("=" * 70)
print("    COMPARATIVE PARKINSONIAN TURNING ANALYSIS (ACTUAL VIDEOS)")
print("=" * 70 + "\n")

N_TRIALS = 200
JITTER_STD = 0.06

for video in sorted(os.listdir(VIDEO_FOLDER)):
    if not video.endswith(".mp4"): continue
    
    print(f"Analyzing {video}...")
    path = os.path.join(VIDEO_FOLDER, video)
    joints = extract_leg_joints(path)
    
    if len(joints) < 30:
        print(f"  → Not enough data to analyze ({len(joints)} frames)\n")
        continue

    features, feature_names = extract_turning_features(joints)
    X_base = np.array(features, dtype=float)
    
    print(f"  → Features Extracted. Running Monte Carlo Simulation ({N_TRIALS} trials per model)...\n")
    
    for name, (model, is_keras) in models_dict.items():
        probs = []
        
        # Monte Carlo Jitter
        for _ in range(N_TRIALS):
            jitter = np.random.normal(1.0, JITTER_STD, X_base.shape)
            X_trial = X_base * jitter
            
            # Formatting input based on model expected format
            if name == "Proposed Bagging":
                # Proposed model expects pandas dataframe with names to match original script
                X_in = pd.DataFrame([X_trial], columns=feature_names)
                p = model.predict_proba(X_in)[0][1]
            elif is_keras:
                # Keras CNNs expect 3D scaled arrays
                X_in = np.expand_dims(scaler.transform([X_trial]), axis=2)
                p = float(model.predict(X_in, verbose=0)[0][0])
            else:
                # Generic sklearn models (ANN)
                X_in = scaler.transform([X_trial])
                p = model.predict_proba(X_in)[0][1]
                
            probs.append(p)
            
        prob = float(np.mean(probs))
        std_score = float(np.std(probs))
        low, high = prob - 1.96 * std_score, prob + 1.96 * std_score
        
        # 3-tier risk label
        if prob >= 0.65:
            risk = "HIGH RISK"
        elif prob >= 0.35:
            risk = "BORDERLINE"
        else:
            risk = "LOW RISK"

        print(f"    - {name.ljust(22)}: {risk.ljust(12)} | Score: {prob:.3f} | 95% CI: [{max(0,low):.3f} - {min(1,high):.3f}] | Uncertainty: {std_score:.3f}")
        
    print("-" * 70)
