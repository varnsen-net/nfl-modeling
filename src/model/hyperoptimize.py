"""Helper functions for optimizing model hyperparameters.

Optimizer: hyperopt"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp
from hyperopt.early_stop import no_progress_loss

from src.model.pipeline import build_swift_pipeline


LIGHTGBM_SPACE = {
    'num_leaves': hp.uniformint('num_leaves', 2, 500),
    'max_depth': hp.uniformint('max_depth', 2, 50),
    'learning_rate': hp.loguniform('learning_rate', -6.0, 0.0),
    'n_estimators': hp.uniformint('n_estimators', 2, 500),
    'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 2, 500),
    'lambda_l1': hp.loguniform('reg_alpha', -6.0, 0.0),
    'lambda_l2': hp.loguniform('reg_lambda', -6.0, 0.0),
    'linear_lambda': hp.loguniform('linear_lambda', -6.0, 0.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 1.0),
    'bagging_freq': hp.uniformint('bagging_freq', 0, 10),
    'feature_fraction': hp.uniform('feature_fraction', 0.2, 1.0),
    'sigmoid': hp.uniform('sigmoid', 0.0, 5.0),
    'verbosity': -1,
    'objective': 'binary',
    'eval_metric': 'binary_logloss',
}


def fix_param_types(params):
    """Convert hyperopt parameters to correct types.

    :param dict params: dictionary of hyperparameters
    :return: dictionary of hyperparameters with correct types
    :rtype: dict
    """
    corrections = {
        'num_leaves': int,
        'max_depth': int,
        'n_estimators': int,
        'min_data_in_leaf': int,
        'min_child_samples': int,
        'bagging_freq': int,
    }
    for key, correction in corrections.items():
        if key in params:
            params[key] = correction(params[key])
    return params


def hyperoptimize(X, y, cv, space, max_evals=100):
    """Optimize hyperparameters using hyperopt.

    :param pd.DataFrame X: features
    :param pd.Series y: target
    :param function cv: cross-validation strategy
    :param dict space: hyperparameter ranges
    :param int max_evals: number of evaluations to perform
    :return: best hyperparameters
    :rtype: dict
    """
    def objective(params):
        """Objective function for hyperopt.

        Uses the SWIFT pipeline and brier loss to evaluate the model.
        
        :param dict params: hyperparameters to test
        :return: loss
        :rtype: float
        """
        print("Testing params:", params)
        pipeline = build_swift_pipeline(params)
        scores = cross_val_score(pipeline, X, y, cv=cv, groups=X['season'],
                                 scoring='neg_brier_score')
        loss = -scores.mean()
        print(f"Loss: {loss}")
        return loss
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        early_stop_fn=no_progress_loss(15),
    )
    best = fix_param_types(best)
    return best
