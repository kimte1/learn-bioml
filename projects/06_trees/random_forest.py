"""
Random forest classifier: bagging (Breiman, 1996, "Bagging Predictors")
plus per-split feature subsampling (Breiman, 2001, "Random Forests") on
top of DecisionTreeClassifier.

Each tree is grown on an independent bootstrap resample of the training
set (sampling n_samples rows *with replacement*), and at every split each
tree only considers a random subset of features (max_features) -- the two
sources of randomness that decorrelate the trees enough for averaging
their predictions to reduce variance without increasing bias much.

Samples never drawn into a given tree's bootstrap ("out-of-bag", OOB) give
that tree a built-in validation set for free -- Breiman's original bagging
insight for estimating generalization error without a held-out split.
"""

from __future__ import annotations

import numpy as np

from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        max_features: int | str | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features is None:
            return n_features
        return self.max_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        n_samples, n_features = X.shape
        self.n_classes_ = int(y.max()) + 1
        max_features = self._resolve_max_features(n_features)
        rng = np.random.default_rng(self.random_state)

        self.trees_: list[DecisionTreeClassifier] = []
        oob_proba_sum = np.zeros((n_samples, self.n_classes_))
        oob_count = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            bootstrap_idx = rng.integers(0, n_samples, size=n_samples)
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[bootstrap_idx] = True
            oob_idx = np.where(~in_bag)[0]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[bootstrap_idx], y[bootstrap_idx])
            self.trees_.append(tree)

            if len(oob_idx) > 0:
                oob_proba_sum[oob_idx] += tree.predict_proba(X[oob_idx])
                oob_count[oob_idx] += 1

        has_oob = oob_count > 0
        if np.any(has_oob):
            oob_proba = oob_proba_sum[has_oob] / oob_count[has_oob, None]
            oob_pred = np.argmax(oob_proba, axis=1)
            self.oob_score_ = float(np.mean(oob_pred == y[has_oob]))
        else:
            self.oob_score_ = None

        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.trees_], axis=0
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([tree.predict_proba(X) for tree in self.trees_], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
