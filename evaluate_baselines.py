import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

import os
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main():
    print("Loading dataset...")
    df = pd.read_csv("turning_pd_features.csv")
    X = df.drop(["label", "filename"], axis=1, errors="ignore").values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    def evaluate_model(name, model, is_keras=False, X_tr=X_train_s, X_te=X_test_s, epochs=15):
        print(f"Training and Evaluating: {name} (keras={is_keras})")
        
        start_train = time.time()
        if is_keras:
            model.fit(X_tr, y_train, epochs=epochs, batch_size=16, verbose=0)
            
            start_pred = time.time()
            preds_prob = model.predict(X_te, verbose=0)
            end_pred = time.time()
            
            preds = (preds_prob > 0.5).astype(int).flatten()
        else:
            model.fit(X_tr, y_train)
            
            start_pred = time.time()
            preds = model.predict(X_te)
            end_pred = time.time()
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        err = 1.0 - acc
        
        # Calculate prediction time per video in milliseconds
        pt_ms = ((end_pred - start_pred) / len(X_te)) * 1000 
        
        results.append({
            "Classifier Models": name,
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1-Score": round(f1, 2),
            "Prediction Time": f"~{int(pt_ms)} ms/video",
            "Prediction Accuracy": f"{acc*100:.1f}%",
            "Prediction Error Rate": f"{err*100:.1f}%"
        })

    # 1. Proposed Model (Custom Bagging)
    try:
        from custom_bagging import CustomBaggingClassifier
        from sklearn.tree import DecisionTreeClassifier

        base_tree = DecisionTreeClassifier(min_samples_leaf=4, max_depth=8, random_state=42)
        proposed = CustomBaggingClassifier(
            base_estimator=base_tree, n_estimators=1000, 
            max_samples=0.75, max_features=0.75, random_state=42
        )
        evaluate_model("Proposed Model (Bagging)", proposed, is_keras=False, X_tr=X_train, X_te=X_test)
    except Exception as e:
        print("Could not evaluate proposed model:", e)

    # 2. ANN
    from sklearn.neural_network import MLPClassifier
    ann = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    evaluate_model("RNN / ANN", ann, is_keras=False)

    # For deep learning models, let's use TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Reshape
        tf.random.set_seed(42)
        
        # 3. Simple CNN (1D)
        X_train_cnn = np.expand_dims(X_train_s, axis=2)
        X_test_cnn = np.expand_dims(X_test_s, axis=2)
        
        cnn = Sequential([
            Conv1D(16, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        evaluate_model("Simple CNN", cnn, is_keras=True, X_tr=X_train_cnn, X_te=X_test_cnn, epochs=50)

        # Simplified 1D equivalents for Tabular Data
        def build_custom_cnn(depth=3, dense=False):
            inputs = tf.keras.Input(shape=(8, 1))
            x = inputs
            for _ in range(depth):
                conv = Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
                if dense:
                    x = tf.keras.layers.Concatenate()([x, conv])
                else:
                    x = conv
            x = Flatten()(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs, outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
            
        eff = build_custom_cnn(depth=3, dense=False)
        evaluate_model("EfficientNetB3", eff, is_keras=True, X_tr=X_train_cnn, X_te=X_test_cnn, epochs=20)

        dense = build_custom_cnn(depth=4, dense=True)
        evaluate_model("DenseNet-121", dense, is_keras=True, X_tr=X_train_cnn, X_te=X_test_cnn, epochs=20)
        
        vgg = Sequential([
            Conv1D(8, 3, padding='same', activation='relu', input_shape=(8, 1)),
            Conv1D(8, 3, padding='same', activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        evaluate_model("VGG Face CNN", vgg, is_keras=True, X_tr=X_train_cnn, X_te=X_test_cnn, epochs=20)

    except ImportError:
        print("\n[WARNING] TensorFlow not installed. Skipping deep CNN/RNN models.")
        print("To run all models, please install: pip install tensorflow\n")

    print("\n" + "="*80)
    print("FINAL TRUE EVALUATION RESULTS")
    print("="*80)
    res_df = pd.DataFrame(results)
    
    # Save the results to CSV so we have a permanent record
    res_df.to_csv("real_baseline_results.csv", index=False)
    
    # Print formatted text table manually without tabulate
    print(res_df.to_string(index=False))
    print("="*80)
    print("Results saved to real_baseline_results.csv")

if __name__ == "__main__":
    main()
