"""
From-scratch second-order gradient boosting in the style of XGBoost
(Chen & Guestrin, 2016, "XGBoost: A Scalable Tree Boosting System"),
built on Friedman's gradient boosting machine (Friedman, 2001, "Greedy
Function Approximation").

Where plain gradient boosting (Friedman, 2001) fits each new tree to the
*first-derivative* residual of the loss, XGBoost fits a regularized
second-order (Newton) approximation: every tree is grown against the
per-sample gradient AND Hessian of the loss, node splits are scored with
a closed-form regularized gain, and each leaf's value is the Newton step
that minimizes that quadratic approximation. Trees are grown level-wise
(breadth-first, bounded by max_depth) -- the point of contrast with
LightGBM's leaf-wise growth in light_gbm.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LAMBDA_EPS = 1e-12


class _Node:
    __slots__ = ("feature_idx", "threshold", "left", "right", "value", "is_leaf")

    def __init__(self) -> None:
        self.feature_idx: int | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.value: float = 0.0
        self.is_leaf: bool = False


@dataclass
class _RegressionTree:
    """A single CART regression tree grown against (gradient, Hessian)
    pairs rather than a target -- the building block XGBoost boosts."""

    max_depth: int = 4
    min_child_weight: float = 1.0  # min sum of Hessians allowed in a leaf
    reg_lambda: float = 1.0  # L2 penalty on leaf weights
    gamma: float = 0.0  # min gain required to keep a split (complexity penalty)
    colsample_bytree: float = 1.0
    random_state: int | None = None

    def fit(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> "_RegressionTree":
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self._rng = np.random.default_rng(self.random_state)
        self._feature_subset = self._candidate_features()
        self.root_ = self._grow(X, grad, hess, depth=0)
        self.n_leaves_, self.depth_reached_ = self._tree_stats(self.root_)
        return self

    def _candidate_features(self) -> np.ndarray:
        if self.colsample_bytree < 1.0:
            k = max(1, int(round(self.colsample_bytree * self.n_features_)))
            return np.sort(self._rng.choice(self.n_features_, size=k, replace=False))
        return np.arange(self.n_features_)

    @staticmethod
    def _leaf_value(G: float, H: float, reg_lambda: float) -> float:
        return -G / (H + reg_lambda + LAMBDA_EPS)

    def _grow(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray, depth: int) -> _Node:
        node = _Node()
        G, H = grad.sum(), hess.sum()
        node.value = self._leaf_value(G, H, self.reg_lambda)

        if depth >= self.max_depth or len(grad) < 2:
            node.is_leaf = True
            return node

        feature_idx, threshold, gain = self._best_split(X, grad, hess, G, H)
        if feature_idx is None or gain <= self.gamma:
            node.is_leaf = True
            return node

        self.feature_importances_[feature_idx] += gain

        left_mask = X[:, feature_idx] <= threshold
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._grow(X[left_mask], grad[left_mask], hess[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], grad[~left_mask], hess[~left_mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray, G_total: float, H_total: float
    ) -> tuple[int | None, float | None, float]:
        """Exact greedy split search (XGBoost's default "exact" algorithm):
        for every feature, sort sample values and sweep the threshold using
        the regularized gain

            gain = 1/2 [ GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam) ] - gamma

        computed from cumulative gradient/Hessian sums, fully vectorized.
        """
        n = len(grad)
        parent_term = (G_total**2) / (H_total + self.reg_lambda + LAMBDA_EPS)

        best_gain, best_feature, best_threshold = 0.0, None, None

        for feature_idx in self._feature_subset:
            values = X[:, feature_idx]
            order = np.argsort(values)
            sorted_values = values[order]
            cum_g = np.cumsum(grad[order])
            cum_h = np.cumsum(hess[order])

            # candidate split sits after index i; only valid where the
            # feature value actually changes (splitting mid-tie is useless)
            distinct = sorted_values[:-1] != sorted_values[1:]
            GL, HL = cum_g[:-1], cum_h[:-1]
            GR, HR = G_total - GL, H_total - HL
            n_left = np.arange(1, n)

            valid = (
                distinct
                & (HL >= self.min_child_weight)
                & (HR >= self.min_child_weight)
                & (n_left >= 1)
                & (n_left <= n - 1)
            )
            if not np.any(valid):
                continue

            gains = 0.5 * (
                GL**2 / (HL + self.reg_lambda + LAMBDA_EPS)
                + GR**2 / (HR + self.reg_lambda + LAMBDA_EPS)
                - parent_term
            )
            gains = np.where(valid, gains, -np.inf)

            i = int(np.argmax(gains))
            if gains[i] > best_gain:
                best_gain = float(gains[i])
                best_feature = feature_idx
                best_threshold = float(sorted_values[i])

        return best_feature, best_threshold, best_gain

    @staticmethod
    def _tree_stats(node: _Node) -> tuple[int, int]:
        """Returns (n_leaves, max depth reached) -- used to contrast
        XGBoost's level-wise growth against LightGBM's leaf-wise growth."""
        if node.is_leaf:
            return 1, 0
        ln, ld = _RegressionTree._tree_stats(node.left)
        rn, rd = _RegressionTree._tree_stats(node.right)
        return ln + rn, 1 + max(ld, rd)

    def _predict_one(self, x: np.ndarray) -> float:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])


@dataclass
class XGBoostRegressor:
    """Additive ensemble of `_RegressionTree`s fit to squared-error
    gradients/Hessians, each tree's contribution shrunk by `learning_rate`
    -- the same shrinkage-then-sum structure as Friedman's gradient
    boosting machine, specialized to XGBoost's regularized, second-order
    tree-building rule.
    """

    n_estimators: int = 150
    learning_rate: float = 0.1
    max_depth: int = 4
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int | None = None

    @staticmethod
    def _grad_hess(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient and Hessian of squared-error loss L = 1/2 (pred - y)^2."""
        return pred - y, np.ones_like(y)

    @staticmethod
    def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> "XGBoostRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.base_score_ = float(y.mean())
        pred = np.full(n_samples, self.base_score_)
        eval_preds = (
            [np.full(len(ey), self.base_score_) for _, ey in eval_set] if eval_set else []
        )

        self.trees_: list[_RegressionTree] = []
        self.feature_importances_ = np.zeros(n_features)
        self.leaves_per_tree_: list[int] = []
        self.depth_per_tree_: list[int] = []
        self.evals_result_: dict[str, list[float]] = (
            {f"eval_{i}": [] for i in range(len(eval_set))} if eval_set else {}
        )

        for _ in range(self.n_estimators):
            grad, hess = self._grad_hess(y, pred)

            if self.subsample < 1.0:
                k = max(2, int(round(self.subsample * n_samples)))
                rows = rng.choice(n_samples, size=k, replace=False)
            else:
                rows = np.arange(n_samples)

            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                colsample_bytree=self.colsample_bytree,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).fit(X[rows], grad[rows], hess[rows])

            pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.feature_importances_ += tree.feature_importances_
            self.leaves_per_tree_.append(tree.n_leaves_)
            self.depth_per_tree_.append(tree.depth_reached_)

            if eval_set:
                for i, (eX, ey) in enumerate(eval_set):
                    eval_preds[i] += self.learning_rate * tree.predict(eX)
                    self.evals_result_[f"eval_{i}"].append(self._rmse(ey, eval_preds[i]))

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        """Yields the running prediction after each additional tree --
        this is what makes a train/test loss-vs-round curve possible."""
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(X)
            yield pred.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred
