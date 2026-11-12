"""
From-scratch gradient boosting in the style of LightGBM (Ke et al.,
2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"),
contrasted here with the XGBoost-style booster in xgb.py. Same Newton
(gradient + Hessian) leaf values and shrinkage as xgb.py, but two
different design choices give LightGBM its speed and its distinct
behavior:

1. Histogram-based splitting: each feature is pre-binned into a fixed
   number of buckets ONCE, up front. Finding the best split at a node
   then means building one small gradient/Hessian histogram per feature
   (vectorized via `np.add.at`) instead of sorting every sample's raw
   value at every node, as the exact-greedy search in xgb.py does.

2. Leaf-wise (best-first) growth instead of level-wise: rather than
   expanding every node at a given depth before going deeper, LightGBM
   always splits whichever *existing leaf* has the single highest gain,
   bounded by `num_leaves` rather than `max_depth`. This tends to reach a
   lower loss for a given leaf budget, but can grow one branch much
   deeper than the rest -- a real overfitting risk on small data that
   level-wise growth avoids by construction.

Gradient-based One-Side Sampling (GOSS) is included as an optional,
toggleable row-subsampling strategy unique to LightGBM: instances with
large gradients (badly fit so far) are always kept, and only a random
fraction of well-fit, small-gradient instances are kept, reweighted to
keep the gradient-sum estimate unbiased.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

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


class _Leaf:
    """A frontier node in the leaf-wise growth queue: holds the sample
    indices it owns, its depth, and its best candidate split."""

    __slots__ = ("node", "idx", "depth", "gain", "feature_idx", "threshold", "left_idx", "right_idx")

    def __init__(self, node: _Node, idx: np.ndarray, depth: int) -> None:
        self.node = node
        self.idx = idx
        self.depth = depth
        self.gain = -np.inf
        self.feature_idx: int | None = None
        self.threshold: float | None = None
        self.left_idx: np.ndarray | None = None
        self.right_idx: np.ndarray | None = None


@dataclass
class _HistogramTree:
    """A regression tree grown leaf-wise over pre-binned features."""

    num_leaves: int = 15
    max_depth: int = -1  # -1 = unbounded, as in real LightGBM
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    max_bin: int = 63

    def fit(self, Xb: np.ndarray, grad: np.ndarray, hess: np.ndarray, bin_edges: list[np.ndarray]) -> "_HistogramTree":
        self._bin_edges = bin_edges
        self.n_features_ = Xb.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)

        root_node = _Node()
        root = _Leaf(root_node, np.arange(len(grad)), depth=0)
        self._score(root, Xb, grad, hess)

        leaves = [root]
        heap: list[tuple[float, int, _Leaf]] = []
        counter = itertools.count()
        if root.gain > 0:
            heapq.heappush(heap, (-root.gain, next(counter), root))

        while heap and len(leaves) < self.num_leaves:
            neg_gain, _, leaf = heapq.heappop(heap)
            if -neg_gain <= 0:
                break

            leaf.node.feature_idx = leaf.feature_idx
            leaf.node.threshold = leaf.threshold
            self.feature_importances_[leaf.feature_idx] += leaf.gain

            left_node, right_node = _Node(), _Node()
            left = _Leaf(left_node, leaf.left_idx, leaf.depth + 1)
            right = _Leaf(right_node, leaf.right_idx, leaf.depth + 1)
            leaf.node.left, leaf.node.right = left_node, right_node

            leaves.remove(leaf)
            leaves.extend([left, right])

            for child in (left, right):
                at_max_depth = self.max_depth >= 0 and child.depth >= self.max_depth
                if at_max_depth or len(child.idx) < 2:
                    continue
                self._score(child, Xb, grad, hess)
                if child.gain > 0:
                    heapq.heappush(heap, (-child.gain, next(counter), child))

        for leaf in leaves:
            G, H = grad[leaf.idx].sum(), hess[leaf.idx].sum()
            leaf.node.value = -G / (H + self.reg_lambda + LAMBDA_EPS)
            leaf.node.is_leaf = True

        self.root_ = root.node
        self.n_leaves_ = len(leaves)
        self.depth_reached_ = max(leaf.depth for leaf in leaves)
        return self

    def _score(self, leaf: _Leaf, Xb: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> None:
        """Build a gradient/Hessian histogram per feature over this leaf's
        samples and find the best bin boundary to split on -- LightGBM's
        histogram algorithm, vectorized with `np.add.at` instead of
        sorting raw feature values (contrast with xgb.py's exact greedy
        search)."""
        idx = leaf.idx
        g, h = grad[idx], hess[idx]
        G_total, H_total = g.sum(), h.sum()
        parent_term = G_total**2 / (H_total + self.reg_lambda + LAMBDA_EPS)

        best_gain, best_feature, best_threshold = 0.0, None, None
        best_left_mask = None

        for feature_idx in range(self.n_features_):
            bins = Xb[idx, feature_idx]
            n_bins = len(self._bin_edges[feature_idx]) + 1

            hist_g = np.zeros(n_bins)
            hist_h = np.zeros(n_bins)
            hist_c = np.zeros(n_bins)
            np.add.at(hist_g, bins, g)
            np.add.at(hist_h, bins, h)
            np.add.at(hist_c, bins, 1)

            cum_g = np.cumsum(hist_g)[:-1]
            cum_h = np.cumsum(hist_h)[:-1]
            cum_c = np.cumsum(hist_c)[:-1]
            GL, HL, nL = cum_g, cum_h, cum_c
            GR, HR = G_total - GL, H_total - HL
            nR = len(idx) - nL

            valid = (nL > 0) & (nR > 0) & (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            if not np.any(valid):
                continue

            gains = 0.5 * (
                GL**2 / (HL + self.reg_lambda + LAMBDA_EPS)
                + GR**2 / (HR + self.reg_lambda + LAMBDA_EPS)
                - parent_term
            )
            gains = np.where(valid, gains, -np.inf)

            b = int(np.argmax(gains))
            if gains[b] > best_gain:
                best_gain = float(gains[b])
                best_feature = feature_idx
                best_threshold = b  # split: bin <= b goes left

        leaf.gain = best_gain
        leaf.feature_idx = best_feature
        leaf.threshold = best_threshold
        if best_feature is not None:
            left_mask = Xb[idx, best_feature] <= best_threshold
            leaf.left_idx = idx[left_mask]
            leaf.right_idx = idx[~left_mask]

    def _predict_one_binned(self, xb: np.ndarray) -> float:
        node = self.root_
        while not node.is_leaf:
            node = node.left if xb[node.feature_idx] <= node.threshold else node.right
        return node.value

    def predict_binned(self, Xb: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one_binned(x) for x in Xb])


@dataclass
class LightGBMRegressor:
    """Additive ensemble of `_HistogramTree`s. Mirrors XGBoostRegressor's
    outer boosting loop (Newton leaves, shrinkage, staged prediction) but
    delegates to leaf-wise, histogram-based tree construction, and
    optionally samples rows per round with GOSS instead of uniform
    `subsample`.
    """

    n_estimators: int = 150
    learning_rate: float = 0.1
    num_leaves: int = 15
    max_depth: int = -1
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    max_bin: int = 63
    use_goss: bool = False
    goss_top_rate: float = 0.2
    goss_other_rate: float = 0.1
    random_state: int | None = None

    @staticmethod
    def _grad_hess(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pred - y, np.ones_like(y)

    @staticmethod
    def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    def _make_bins(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Quantile-bin every feature once, up front, over the full
        training set -- the preprocessing step that makes histogram
        splitting possible."""
        n_features = X.shape[1]
        bin_edges = []
        Xb = np.empty(X.shape, dtype=np.int32)
        quantiles = np.linspace(0, 100, self.max_bin + 1)[1:-1]
        for f in range(n_features):
            edges = np.unique(np.percentile(X[:, f], quantiles))
            bin_edges.append(edges)
            Xb[:, f] = np.searchsorted(edges, X[:, f], side="right")
        return Xb, bin_edges

    def _goss_sample(
        self, grad: np.ndarray, hess: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Gradient-based One-Side Sampling: always keep the top_rate
        fraction of instances with the largest |gradient| (the ones the
        ensemble still fits poorly), randomly keep an other_rate fraction
        of the rest, and amplify that sampled remainder by
        (1 - top_rate) / other_rate so the gradient-sum estimate used for
        splitting stays unbiased."""
        n = len(grad)
        order = np.argsort(-np.abs(grad))
        n_top = int(np.ceil(self.goss_top_rate * n))
        top_idx = order[:n_top]
        rest = order[n_top:]

        n_rand = int(np.ceil(self.goss_other_rate * n))
        n_rand = min(n_rand, len(rest))
        sampled_rest = rng.choice(rest, size=n_rand, replace=False)

        amplify = (1.0 - self.goss_top_rate) / max(self.goss_other_rate, 1e-12)
        idx = np.concatenate([top_idx, sampled_rest])
        g_used = np.concatenate([grad[top_idx], grad[sampled_rest] * amplify])
        h_used = np.concatenate([hess[top_idx], hess[sampled_rest] * amplify])
        used_fraction = len(idx) / n
        return idx, g_used, h_used, used_fraction

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> "LightGBMRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        Xb, self._bin_edges = self._make_bins(X)

        self.base_score_ = float(y.mean())
        pred = np.full(n_samples, self.base_score_)
        eval_preds = (
            [np.full(len(ey), self.base_score_) for _, ey in eval_set] if eval_set else []
        )

        self.trees_: list[_HistogramTree] = []
        self.feature_importances_ = np.zeros(n_features)
        self.leaves_per_tree_: list[int] = []
        self.depth_per_tree_: list[int] = []
        self.used_fraction_per_round_: list[float] = []
        self.evals_result_: dict[str, list[float]] = (
            {f"eval_{i}": [] for i in range(len(eval_set))} if eval_set else {}
        )

        eval_Xb_full = (
            [self._bin_features(ex) for ex, _ in eval_set] if eval_set else []
        )

        for _ in range(self.n_estimators):
            grad, hess = self._grad_hess(y, pred)

            if self.use_goss:
                idx, g_used, h_used, used_fraction = self._goss_sample(grad, hess, rng)
            else:
                idx = np.arange(n_samples)
                g_used, h_used = grad, hess
                used_fraction = 1.0
            self.used_fraction_per_round_.append(used_fraction)

            tree = _HistogramTree(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                max_bin=self.max_bin,
            ).fit(Xb[idx], g_used, h_used, self._bin_edges)

            pred += self.learning_rate * tree.predict_binned(Xb)
            self.trees_.append(tree)
            self.feature_importances_ += tree.feature_importances_
            self.leaves_per_tree_.append(tree.n_leaves_)
            self.depth_per_tree_.append(tree.depth_reached_)

            if eval_set:
                for i, (_, ey) in enumerate(eval_set):
                    eval_preds[i] += self.learning_rate * tree.predict_binned(eval_Xb_full[i])
                    self.evals_result_[f"eval_{i}"].append(self._rmse(ey, eval_preds[i]))

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _bin_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xb = np.empty(X.shape, dtype=np.int32)
        for f, edges in enumerate(self._bin_edges):
            Xb[:, f] = np.searchsorted(edges, X[:, f], side="right")
        return Xb

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self._bin_features(X)
        pred = np.full(Xb.shape[0], self.base_score_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict_binned(Xb)
            yield pred.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self._bin_features(X)
        pred = np.full(Xb.shape[0], self.base_score_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict_binned(Xb)
        return pred
