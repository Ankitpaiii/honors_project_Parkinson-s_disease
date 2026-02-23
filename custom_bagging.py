import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

class CustomBaggingClassifier:
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=0.8, random_state=None):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features  # Fraction of features to subset
        self.random_state = random_state
        self.estimators_ = []
        self.feature_indices_ = []  # Store which features each tree saw

    def fit(self, X, y):
        self.estimators_ = []
        self.feature_indices_ = []
        
        # Convert to numpy for indexing
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        n_sub_samples = int(self.max_samples * n_samples)
        # Ensure at least 1 feature is selected
        n_sub_features = max(1, int(self.max_features * n_features))
        
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        
        for i in range(self.n_estimators):
            # 1. Row Bootstrapping (Bagging)
            indices = rng.choice(n_samples, n_sub_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # 2. Feature Subsampling (Random Forest improvement)
            feat_indices = rng.choice(n_features, n_sub_features, replace=False)
            X_sample = X_sample[:, feat_indices]
            
            # 3. Training
            estimator = clone(self.base_estimator)
            if hasattr(estimator, "random_state"):
                estimator.random_state = rng.randint(0, 2**31 - 1)
            
            estimator.fit(X_sample, y_sample)
            
            self.estimators_.append(estimator)
            self.feature_indices_.append(feat_indices)
            
        return self

    def predict_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        combined_probs = np.zeros((n_samples, n_classes))
        
        for estimator, feat_indices in zip(self.estimators_, self.feature_indices_):
            # Each estimator ONLY sees its assigned feature columns
            X_subset = X[:, feat_indices]
            proba = estimator.predict_proba(X_subset)
            
            # Map probabilities to global classes
            if proba.shape[1] == n_classes:
                combined_probs += proba
            else:
                for i, cls in enumerate(estimator.classes_):
                    global_idx = np.where(self.classes_ == cls)[0][0]
                    combined_probs[:, global_idx] += proba[:, i]
        
        return combined_probs / len(self.estimators_)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
