"""Helper functions for optimizing model hyperparameters.

Optimizer: hyperopt"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp

from src.model.pipeline import build_swift_pipeline


LIGHTGBM_SPACE = {
    'num_leaves': hp.uniformint('num_leaves', 2, 500),
    'max_depth': hp.uniformint('max_depth', -1, 500),
    'learning_rate': hp.loguniform('learning_rate', -6.0, 0.0),
    'n_estimators': hp.uniformint('n_estimators', 2, 500),
    'reg_alpha': hp.loguniform('reg_alpha', -6.0, 0.0),
    'reg_lambda': hp.loguniform('reg_lambda', -6.0, 0.0),
    'verbosity': -1,
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
    }
    for key, correction in corrections.items():
        if key in params:
            params[key] = correction(params[key])
    return params


def hyperoptimize(X, y, space, max_evals=100):
    """Optimize hyperparameters using hyperopt.

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
        scores = cross_val_score(pipeline, X, y, cv=7, scoring='neg_brier_score')
        loss = -scores.mean()
        print(f"Loss: {loss}")
        return loss
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals
    )
    best = fix_param_types(best)
    return best
