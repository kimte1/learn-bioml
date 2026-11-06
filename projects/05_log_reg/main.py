"""
Binary logistic regression from scratch, cross-checked against scikit-learn.


Where linear regression predicts a continuous number, logistic regression
predicts a probability: it applies the sigmoid function to the same
w.x + b linear score, squashing it into (0, 1). It's trained by minimizing
cross-entropy loss (not mean squared error) via the same batch gradient
descent used in 01_linear_reg -- only the loss function and its gradient
change, the optimization loop is identical in spirit.

Binary logistic regression is the two-class special case of softmax
regression: softmax([z, 0]) reduces exactly to sigmoid(z) (verified
numerically in main()). Softmax is what you'd reach for instead if there
were more than two classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from plotting import (
    plot_class_imbalance_effect,
    plot_coef_comparison,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_precision_recall_f1_vs_threshold,
    plot_roc_comparison,
    plot_sigmoid,
    plot_weight_evolution,
)

FIG_DIR = Path(__file__).resolve().parent / "fig"



def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.3, 
        help="Gradient descent learning rate"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=500, 
        help="Number of gradient descent epochs"
)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=25,
        help="Record weights every N epochs for plotting",
    )
    parser.add_argument(
        "--minority-keep-frac",
        type=float,
        default=0.12,
        help="Fraction of malignant (minority) cases kept when constructing the imbalanced dataset",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=3, 
        help="Random seed for splits and subsampling"
    )
    return parser.parse_args()



def standardize_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std




def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row max for numerical stability.
    Generalizes sigmoid to more than two classes -- see module docstring.
    """
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)



def sigmoid(z: np.ndarray) -> np.ndarray:
    """ 1 / (1 + e^-z) """

    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )



def cross_entropy_loss(
    y: np.ndarray,
    p: np.ndarray,
    sample_weight: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Weighted mean binary cross-entropy: -mean[w * (y*log(p) + (1-y)*log(1-p))].
    p is clipped away from exactly 0 or 1 so log() never hits -inf.
    """
    p_clipped = np.clip(p, eps, 1 - eps)
    per_sample = -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    return float(np.sum(sample_weight * per_sample) / np.sum(sample_weight))



def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    sample_weight: np.ndarray | None = None,
    snapshot_interval: int = 25,
) -> tuple[np.ndarray, float, list[float], np.ndarray, np.ndarray]:
    """
    train using batch gradient descent on cross-entropy loss
    """
    n_samples, n_features = X.shape

    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    weight_sum = sample_weight.sum()

    w = np.zeros(n_features)
    b = 0.0
    loss_history: list[float] = []
    weight_history: list[np.ndarray] = []
    bias_history: list[float] = []
    
    for epoch in range(epochs):
        p = sigmoid(X @ w + b)
        loss = cross_entropy_loss(y, p, sample_weight)

        loss_history.append(loss)
        weight_history.append(w.copy())
        bias_history.append(b)

        error = p - y
        grad_w = (X.T @ (sample_weight * error)) / weight_sum
        grad_b = np.sum(sample_weight * error) / weight_sum
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, loss_history, np.array(weight_history), np.array(bias_history)




def predict_proba(
    X: np.ndarray, 
    w: np.ndarray, 
    b: float
) -> np.ndarray:

    return sigmoid(X @ w + b)




def confusion_matrix_counts(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> np.ndarray:
    """Returns [[TN, FP], [FN, TP]]."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])



def roc_curve_scratch(
    y_true: np.ndarray, 
    scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the classification threshold over every distinct predicted score,
    from "classify everything positive" down to "classify everything
    negative," recording (FPR, TPR) at each one. This is exactly what
    sklearn.metrics.roc_curve does internally.
    """
    thresholds = np.unique(scores)[::-1]
    thresholds = np.concatenate(([thresholds[0] + 1e-6], thresholds, [thresholds[-1] - 1e-6]))

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    fpr = np.zeros(len(thresholds))
    tpr = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr[i] = tp / n_pos if n_pos > 0 else 0.0
        fpr[i] = fp / n_neg if n_neg > 0 else 0.0

    return fpr, tpr, thresholds



def precision_recall_f1(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall) 
    else:
       f1 = 0.0

    accuracy = (tp + tn) / cm.sum()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }



def precision_recall_curve_scratch(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the classification threshold over every distinct predicted score
    (same sweep as roc_curve_scratch) and compute precision/recall/F1 at
    each one, to see the tradeoff instead of just its value at threshold=0.5.
    """
    thresholds = np.unique(scores)[::-1]
    thresholds = np.concatenate(([thresholds[0] + 1e-6], thresholds, [thresholds[-1] - 1e-6]))

    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))
    f1 = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        y_pred = (scores >= t).astype(int)
        cm = confusion_matrix_counts(y_true, y_pred)
        metrics = precision_recall_f1(cm)
        precision[i] = metrics["precision"]
        recall[i] = metrics["recall"]
        f1[i] = metrics["f1"]

    return thresholds, precision, recall, f1



def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve via the trapezoidal rule. fpr/tpr must already
    be in ascending order (as roc_curve_scratch returns them, by sweeping
    thresholds from high to low) -- re-sorting here would be a no-op at best,
    and at worst an unstable sort could scramble the pairing between tied
    fpr values and their tpr values, corrupting the integral.
    """
    return float(np.trapezoid(tpr, fpr))


def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    #### load data and prep
    data = load_breast_cancer()
    
    X_all = data.data # list of lists
    y_all = 1 - data.target # 1 (malignant) or 0 (not malignant)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, 
        y_all, 
        test_size=0.2, 
        shuffle=True,
        stratify=y_all, 
        random_state=args.seed,
    )

    X_train_std, train_mean, train_scale = standardize_matrix(X_train)
    X_test_std = (X_test - train_mean) / train_scale

    ### Train from scratch via batch gradient descent on cross-entropy loss.
    w, b, loss_history, weight_history, bias_history = train_logistic_regression(
        X_train_std,
        y_train,
        lr=args.lr,
        epochs=args.epochs,
        snapshot_interval=args.snapshot_interval,
    )
    plot_loss_curve(loss_history, FIG_DIR / "01_loss_vs_epoch.png")

    
    plot_weight_evolution(
        np.arange(0, args.epochs, 1), 
        weight_history, 
        bias_history, 
        list(data.feature_names), 
        FIG_DIR / "02_weight_evolution.png",
    )

    ### Evaluate model
    # pass standardized test set through sigmoid (which gives probabilities)
    test_proba = predict_proba(X_test_std, w, b)
    # make prediction
    test_pred = (test_proba >= 0.5).astype(int)

    # plot confusion matrix
    cm = confusion_matrix_counts(y_test, test_pred)
    plot_confusion_matrix(
        cm, 
        ["benign", "malignant"], 
        "From-scratch logistic regression (threshold=0.5)",
        FIG_DIR / "03_confusion_matrix.png",
    )

    # plot ROC (compare scratch vs sklearn's ROC)    
    fpr_scratch, tpr_scratch, _ = roc_curve_scratch(y_test, test_proba)
    auc_scratch = auc_trapezoid(fpr_scratch, tpr_scratch)

    sk_model = LogisticRegression(max_iter=10000).fit(X_train_std, y_train)
    sk_proba = sk_model.predict_proba(X_test_std)[:, 1]
    fpr_sklearn, tpr_sklearn, _ = roc_curve_scratch(y_test, sk_proba)
    auc_sklearn = auc_trapezoid(fpr_sklearn, tpr_sklearn)
    auc_sklearn_builtin = roc_auc_score(y_test, sk_proba)
    
    plot_roc_comparison(
        fpr_scratch,
        tpr_scratch,
        auc_scratch,
        fpr_sklearn,
        tpr_sklearn,
        auc_sklearn,
        auc_sklearn_builtin,
        FIG_DIR / "04_roc_curve_comparison.png",
    )


    metrics = precision_recall_f1(cm)

    thresholds, precisions, recalls, f1s = precision_recall_curve_scratch(y_test, test_proba)
    plot_precision_recall_f1_vs_threshold(
        thresholds,
        precisions,
        recalls,
        f1s,
        FIG_DIR / "05_precision_recall_f1_vs_threshold.png",
    )


if __name__ == "__main__":
    main()
