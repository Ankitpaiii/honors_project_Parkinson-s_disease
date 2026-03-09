import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin


class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom Bagging Ensemble Classifier for Parkinson's Disease Detection.

    Combines row bootstrapping (bagging) with feature sub-sampling to build
    an ensemble of diverse decision trees. Each tree votes probabilistically,
    producing a smooth confidence score between 0 and 1 — critical for
    clinical applications where uncertainty matters.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=DecisionTreeClassifier()
    n_estimators   : int, number of trees in the ensemble
    max_samples    : float, fraction of training samples per tree (bagging)
    max_features   : float, fraction of features per tree (sub-spacing)
    random_state   : int, for reproducibility
    """

    def __init__(self, base_estimator=None, n_estimators=100,
                 max_samples=0.75, max_features=0.75, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators   = n_estimators
        self.max_samples    = max_samples
        self.max_features   = max_features
        self.random_state   = random_state

    # ── Sklearn interface ──────────────────────────────────────────────────────
    def get_params(self, deep=True):
        return {
            "base_estimator": self.base_estimator,
            "n_estimators":   self.n_estimators,
            "max_samples":    self.max_samples,
            "max_features":   self.max_features,
            "random_state":   self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return (f"CustomBaggingClassifier("
                f"n_estimators={self.n_estimators}, "
                f"max_samples={self.max_samples}, "
                f"max_features={self.max_features}, "
                f"random_state={self.random_state})")

    # ── Training ───────────────────────────────────────────────────────────────
    def fit(self, X, y):
        _base = self.base_estimator if self.base_estimator is not None \
                else DecisionTreeClassifier()

        self.estimators_     = []
        self.feature_indices_ = []

        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        n_sub_samples  = int(self.max_samples * n_samples)
        n_sub_features = max(1, int(self.max_features * n_features))

        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)

        for _ in range(self.n_estimators):
            # 1. Row bootstrap (bagging)
            idx = rng.choice(n_samples, n_sub_samples, replace=True)
            X_s = X[idx]
            y_s = y[idx]

            # 2. Feature subspace (random subspace method)
            feat_idx = rng.choice(n_features, n_sub_features, replace=False)
            X_s = X_s[:, feat_idx]

            # 3. Fit a fresh clone
            est = clone(_base)
            if hasattr(est, "random_state"):
                est.random_state = rng.randint(0, 2**31 - 1)
            est.fit(X_s, y_s)

            self.estimators_.append(est)
            self.feature_indices_.append(feat_idx)

        return self

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict_proba(self, X):
        X = np.array(X)
        n_samples  = X.shape[0]
        n_classes  = len(self.classes_)
        probs      = np.zeros((n_samples, n_classes))

        for est, feat_idx in zip(self.estimators_, self.feature_indices_):
            p = est.predict_proba(X[:, feat_idx])
            if p.shape[1] == n_classes:
                probs += p
            else:
                for i, cls in enumerate(est.classes_):
                    gi = np.where(self.classes_ == cls)[0][0]
                    probs[:, gi] += p[:, i]

        return probs / self.n_estimators

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
