"""
Decision tree classifier (CART-style, Breiman, Friedman, Olshen & Stone,
1984), grown by greedy recursive binary splitting: at every node, try
every (feature, threshold) pair and keep whichever split most reduces
class impurity (Gini or entropy), weighted by how samples divide between
the two children.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _gini(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


_CRITERIA = {"gini": _gini, "entropy": _entropy}


class Node:
    __slots__ = ("feature_idx", "threshold", "left", "right", "proba", "is_leaf")

    def __init__(self) -> None:
        self.feature_idx: int | None = None
        self.threshold: float | None = None
        self.left: "Node | None" = None
        self.right: "Node | None" = None
        self.proba: np.ndarray | None = None
        self.is_leaf: bool = False



@dataclass
class DecisionTreeClassifier:
    """
    Binary-split classification tree. 

    Growth stops at a node once it's
    pure, too small to split further (min_samples_split), or at
    max_depth -- or once no candidate split reduces impurity at all.

    """

    max_depth: int | None = None # how many splits deep the tree can grow
    min_samples_split: int = 2 # node won't be split further it has fewer samples than 2
    min_samples_leaf: int = 1 # candidate split is rejected if child would have fewer samples than 1
    criterion: str = "gini" # 
    max_features: int | None = None # max number of features to consider, if not None
    random_state: int | None = None

    def __post_init__(self) -> None:
        self._impurity = _CRITERIA[self.criterion]

    # entry point
    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:

        # validate inputs
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # init bookkeeping
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self._rng = np.random.default_rng(self.random_state)

        # grow nodes
        self.root_ = self._grow(X, y, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _class_counts(self, y: np.ndarray) -> np.ndarray:
        return np.bincount(y, minlength=self.n_classes_).astype(float)

    def _grow(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        depth: int
    ) -> Node:
    
        node = Node()

        # count how many samples belong to each class
        counts = self._class_counts(y)

        # add proba in class node [P(class 0), P(class 1)]
        # this is the % of class 0 and % of class 1
        node.proba = counts / counts.sum()

        # check that the node is already classified (no reason to split)
        is_pure = counts.max() == counts.sum()

        # check if current node hit the depth limit
        at_max_depth = self.max_depth is not None and depth >= self.max_depth

        # check if node should stop growing and become a leaf
        if is_pure or len(y) < self.min_samples_split or at_max_depth:
            node.is_leaf = True
            return node

        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None or gain <= 0:
            node.is_leaf = True
            return node

        self.feature_importances_[feature_idx] += gain * len(y)

        left_mask = X[:, feature_idx] <= threshold

        assert 0 < left_mask.sum() < len(y), (
            f"Invalid split: feature={feature_idx}, "
            f"threshold={threshold}, "
            f"left={left_mask.sum()}, right={(~left_mask).sum()}"
        ) 
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node


    
 
    def _candidate_features(self) -> np.ndarray:
        if self.max_features is not None and self.max_features < self.n_features_:
            return self._rng.choice(self.n_features_, size=self.max_features, replace=False)
        return np.arange(self.n_features_)


    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None, float]:
        """
        Scan every candidate feature's sorted values for the split that
        most reduces weighted impurity. Class counts on each side of a
        candidate split are read off a running cumulative sum (one pass
        per feature) rather than recounted from scratch at every threshold.
        """
        n_samples = len(y)
        y_onehot = np.zeros((n_samples, self.n_classes_))
        y_onehot[np.arange(n_samples), y] = 1

        parent_impurity = self._impurity(y_onehot.sum(axis=0))

        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature_idx in self._candidate_features():
            values = X[:, feature_idx]
            order = np.argsort(values)
            sorted_values = values[order]
            cum_counts = np.cumsum(y_onehot[order], axis=0)
            total_counts = cum_counts[-1]

            distinct_value_boundary = sorted_values[:-1] != sorted_values[1:]
            for i in np.where(distinct_value_boundary)[0]:
                left_n, right_n = i + 1, n_samples - (i + 1)
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                left_counts = cum_counts[i]
                right_counts = total_counts - left_counts
                weighted_impurity = (
                    left_n / n_samples * self._impurity(left_counts)
                    + right_n / n_samples * self._impurity(right_counts)
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(sorted_values[i])

        return best_feature, best_threshold, best_gain

    
    
    def _predict_one(self, x: np.ndarray) -> np.ndarray:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    # entry for prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
