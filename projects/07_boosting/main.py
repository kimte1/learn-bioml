"""
- Build gradient boosting from-scratch: XGBoost (xgb.py) and LightGBM (light_gbm.py) regressors

- model predicts protein melting temperature (Tm) from sequence-derived features, using
Meltome Atlas (Jarzab et al., 2020, Nature Methods) thermal proteome stability dataset.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from data_load_prep import make_protein_tm_dataset
from light_gbm import LightGBMRegressor
from plotting import (
    plot_feature_importance,
    plot_goss_effect,
    plot_loss_curves,
    plot_model_comparison,
    plot_pred_vs_actual,
    plot_tree_growth_strategy,
)
from xgb import XGBoostRegressor

FIG_DIR = Path(__file__).resolve().parent / "fig"



def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-estimators", 
        type=int, 
        default=150, 
        help="Boosting rounds for XGBoost/LightGBM"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.1, 
        help="Shrinkage applied to each tree"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed for data generation and model fitting"
    )
    return parser.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    X, y, feature_names, gene_names = make_protein_tm_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    ### Train XGBoost and LightGBM, tracking test-set loss every round
    xgb_model = XGBoostRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=4,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=args.seed,
    ).fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    lgbm_model = LightGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=15,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=args.seed,
    ).fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    xgb_pred = xgb_model.predict(X_test)
    lgbm_pred = lgbm_model.predict(X_test)

    ### scikit-learn cross-check, as in every prior week's project
    sklearn_model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=4,
        random_state=args.seed,
    ).fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)


    ### Figure 1: loss (RMSE) vs. boosting round, train and test, both models
    rounds = np.arange(1, args.n_estimators + 1)
    plot_loss_curves(
        rounds,
        np.array(xgb_model.evals_result_["eval_0"]),
        np.array(xgb_model.evals_result_["eval_1"]),
        np.array(lgbm_model.evals_result_["eval_0"]),
        np.array(lgbm_model.evals_result_["eval_1"]),
        FIG_DIR / "01_loss_curves.png",
    )

    ### Figure 2: predicted vs. actual Tm on the held-out test set
    plot_pred_vs_actual(y_test, xgb_pred, lgbm_pred, FIG_DIR / "02_pred_vs_actual.png")

    ### Figure 3: feature importance, XGBoost vs. LightGBM
    plot_feature_importance(
        feature_names, xgb_model.feature_importances_, lgbm_model.feature_importances_,
        FIG_DIR / "03_feature_importance.png",
    )

    ### Figure 4: final test accuracy across all three models
    plot_model_comparison(
        ["XGBoost (scratch)", "LightGBM (scratch)", "sklearn GBR"],
        [rmse(y_test, xgb_pred), rmse(y_test, lgbm_pred), rmse(y_test, sklearn_pred)],
        [r_squared(y_test, xgb_pred), r_squared(y_test, lgbm_pred), r_squared(y_test, sklearn_pred)],
        FIG_DIR / "04_model_comparison.png",
    )

    ### Figure 5: level-wise (XGBoost) vs. leaf-wise (LightGBM) tree growth
    plot_tree_growth_strategy(
        np.array(xgb_model.leaves_per_tree_),
        np.array(xgb_model.depth_per_tree_),
        np.array(lgbm_model.leaves_per_tree_),
        np.array(lgbm_model.depth_per_tree_),
        FIG_DIR / "05_tree_growth_strategy.png",
    )

    ### Figure 6: GOSS -- LightGBM trained on a gradient-biased data subset
    ### per round vs. the same model trained on the full data every round
    lgbm_no_goss = LightGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=15,
        reg_lambda=1.0,
        min_child_weight=1.0,
        use_goss=False,
        random_state=args.seed,
    ).fit(X_train, y_train, eval_set=[(X_train, y_train)])

    lgbm_goss = LightGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=15,
        reg_lambda=1.0,
        min_child_weight=1.0,
        use_goss=True,
        goss_top_rate=0.2,
        goss_other_rate=0.1,
        random_state=args.seed,
    ).fit(X_train, y_train, eval_set=[(X_train, y_train)])

    plot_goss_effect(
        rounds,
        np.array(lgbm_no_goss.evals_result_["eval_0"]),
        np.array(lgbm_goss.evals_result_["eval_0"]),
        np.array(lgbm_goss.used_fraction_per_round_),
        FIG_DIR / "06_goss_effect.png",
    )


if __name__ == "__main__":
    main()
