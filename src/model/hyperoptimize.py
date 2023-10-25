"""Helper functions for optimizing model hyperparameters.

Optimizer: hyperopt"""

from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, space_eval
from hyperopt.early_stop import no_progress_loss

from src.config.config import DEFAULT_PARAM_PREFIX


def map_name_to_param(param, prefix):
    """Map a hyperparameter name to its distribution or value.
    
    :param namedtuple param: a named tuple with the following fields:
        name, value, min, max
    :param str prefix: the prefix to add to the key
    :return: a key-value pair for the hyperparameter
    :rtype: tuple
    """
    key = f"{prefix}{param.name}"
    if param.min is not None and param.max is not None:
        return (key, param.value(key, param.min, param.max))
    else:
        return (key, param.value)


def map_name_to_type(param, prefix):
    """Map a hyperparameter name to its type.
    
    :param namedtuple param: a named tuple with the following fields:
        name, type
    :param str prefix: the prefix to add to the key
    :return: a key-value pair for the parameter type
    :rtype: tuple
    """
    key = f"{prefix}{param.name}"
    return (key, param.type)


def make_param_mapping(params, mapper, prefix=DEFAULT_PARAM_PREFIX):
    """Make a parameter space from a list of named tuples.
    
    :param list[namedtuple] params: named tuples with the following fields:
        name, value, min, max, type
    :param callable mapper: a function that converts a named tuple to a key-value pair
    :param str prefix: the prefix to add to the key
    :return: hyperparameter space
    :rtype: dict
    """
    param_space = {}
    for param in params:
        key, value = mapper(param, prefix)
        param_space.update({key: value})
    return param_space


def crossval_objective(params, model, X, y, cv, scoring):
    """Objective function for hyperopt.

    Uses cross validation scoring.
    
    :param dict params: hyperparameters to test
    :param sklearn.base.BaseEstimator model: estimator to test
    :param pd.DataFrame X: features
    :param pd.Series y: target
    :param Union[int, callable] cv: cross-validation strategy
    :param Union[str, dict] scoring: scoring metric
    :return: loss
    :rtype: float
    """
    print("Testing params:", params)
    model.set_params(**params)
    scores = cross_val_score(model, X, y, cv=cv, groups=X['season'],
                             scoring=scoring)
    loss = -scores.mean()
    print(f"Loss: {loss}")
    return loss


def find_best_params(objective, space, max_evals, early_stop_n):
    """Search for the optimal model hyperparams using hyperopt.

    :param callable objective: objective function to minimize
    :param dict space: hyperparameter ranges
    :param int max_evals: number of evaluations to perform
    :param int early_stop_n: number of iterations without improvement to stop
    :return: best hyperparameters
    :rtype: dict
    """
    best_params = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        early_stop_fn=no_progress_loss(early_stop_n),
    )
    return best_params


def fix_param_dtypes(params, param_dtypes):
    """Fix types for best hyperparameters returned by hyperopt."""
    for key, dtype in param_dtypes.items():
        val = dtype(params.get(key))
        params.update({key: val})
    return params


def hyperoptimize(model, X, y, cv, scoring, space, objective=crossval_objective,
                  max_evals=100, early_stop_n=15):
    """Optimize model hyperparameters.

    :param sklearn.base.BaseEstimator model: estimator to test
    :param callable objective: objective function to minimize
    :param pd.DataFrame X: features
    :param pd.Series y: target
    :param Union[int, callable] cv: cross-validation strategy
    :param Union[str, dict] scoring: scoring metric
    :param list[namedtuple] space: named tuples with the following fields:
        name, value, min, max, type
    :param int max_evals: max number of evaluations to perform
    :param int early_stop_n: number of iterations without improvement to stop
    :return: best hyperparameters
    :rtype: dict
    """
    search_space = make_param_mapping(space, map_name_to_param)
    objective = partial(objective, model=model, X=X, y=y, cv=cv,
                        scoring=scoring)
    best_params = find_best_params(objective,
                                   search_space,
                                   max_evals=max_evals,
                                   early_stop_n=early_stop_n)
    best_params = space_eval(search_space, best_params)
    return best_params
