"""
Decision trees and random forests from scratch, cross-checked against
scikit-learn, on a synthetic protein-solubility classification task.

A decision tree and a random forest are genuinely different algorithms,
not variations of one: a tree is a single greedy recursive splitter
(decision_tree.py), while a forest is an ensemble of many such trees,
each grown on a bootstrap resample of the data with per-split feature
subsampling (random_forest.py) -- bagging and feature
subsampling decorrelates the trees enough for
averaging them to reduce variance.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

from data_load_prep import make_protein_dataset

from plotting import (
    plot_bagging_variance_reduction,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_accuracy_comparison,
    plot_overfitting_diagnostic,
)


FIG_DIR = Path(__file__).resolve().parent / "fig"


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--max-depth-sweep",
        type=int,
        default=15,
        help="Largest max_depth tried in the single-tree overfitting diagnostic",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the main random forest model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for model fitting",
    )

    return parser.parse_args()



def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))





def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    X, y, feature_names = make_protein_dataset(
        n_samples=1000, 
        seed=0, 
        noise_std=1.0,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=0, 
        stratify=y,
        shuffle=True,
    )

    ### overfitting diagnostic
    # train a fresh decision tree at each max depth
    # record train and test accuracy at each depth
    max_depths = np.arange(1, args.max_depth_sweep + 1)
    train_acc = np.zeros(len(max_depths))
    test_acc = np.zeros(len(max_depths))

    for i, depth in enumerate(max_depths):
        tree = DecisionTreeClassifier(
            max_depth=int(depth), 
            random_state=args.seed
        ).fit(X_train, y_train)

        train_acc[i] = accuracy(y_train, tree.predict(X_train))
        test_acc[i] = accuracy(y_test, tree.predict(X_test))

    plot_overfitting_diagnostic(
        max_depths, 
        train_acc, 
        test_acc, 
        FIG_DIR / "01_overfitting_diagnostic.png"
    )
    
    ### Feature importance: single tree (unrestricted depth) vs. forest
    full_tree = DecisionTreeClassifier(
        random_state=args.seed
    ).fit(X_train, y_train)

    forest = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        random_state=args.seed
    ).fit(X_train, y_train)

    plot_feature_importance(
        feature_names,
        full_tree.feature_importances_,
        forest.feature_importances_,
        FIG_DIR / "02_feature_importance.png",
    )

    ### Bagging via bootstrap resampling: forest test error vs. n_estimators,
    ### against the spread of many independently bootstrapped single trees
    n_estimators_list = np.array([1, 2, 5, 10, 20, 50, 100, 150, 200])
    forest_test_error = np.zeros(len(n_estimators_list))
    for i, n_estimators in enumerate(n_estimators_list):
        rf = RandomForestClassifier(
            n_estimators=int(n_estimators), 
            random_state=args.seed
        ).fit(X_train, y_train)

        forest_test_error[i] = 1 - accuracy(y_test, rf.predict(X_test))

    n_bootstrap_trees = 30
    rng = np.random.default_rng(args.seed)
    single_tree_errors = np.zeros(n_bootstrap_trees)
    for i in range(n_bootstrap_trees):
        boot_idx = rng.integers(0, len(X_train), size=len(X_train))
        tree = DecisionTreeClassifier(
            random_state=int(rng.integers(0, 2**31 - 1))
        ).fit(
            X_train[boot_idx], 
            y_train[boot_idx],
        )
        single_tree_errors[i] = 1 - accuracy(y_test, tree.predict(X_test))

    plot_bagging_variance_reduction(
        n_estimators_list, 
        forest_test_error, 
        single_tree_errors, 
        FIG_DIR / "03_bagging_variance_reduction.png"
    )
    
    ### Cross-check against scikit-learn
    sk_tree = SklearnDecisionTreeClassifier(
        random_state=args.seed
    ).fit(X_train, y_train)
    sk_forest = SklearnRandomForestClassifier(
        n_estimators=args.n_estimators, 
        random_state=args.seed,
    ).fit(X_train, y_train)

    model_labels = [
        "Scratch tree", 
        "sklearn tree", 
        "Scratch forest", 
        "sklearn forest"
    ]
    model_accuracies = [
        accuracy(y_test, full_tree.predict(X_test)),
        accuracy(y_test, sk_tree.predict(X_test)),
        accuracy(y_test, forest.predict(X_test)),
        accuracy(y_test, sk_forest.predict(X_test)),
    ]
    plot_model_accuracy_comparison(
        model_labels, 
        model_accuracies, 
        FIG_DIR / "04_model_accuracy_comparison.png"
    )
    
    ### Confusion matrix for the final scratch forest
    cm = confusion_matrix(y_test, forest.predict(X_test))
    plot_confusion_matrix(
        cm,
        ["insoluble", "soluble"],
        "From-scratch random forest",
        FIG_DIR / "05_confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
