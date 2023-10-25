"""Hyperparameter spaces."""

from collections import namedtuple

from hyperopt import hp


param = namedtuple('param', ['name', 'value', 'min', 'max', 'type'])

LIGHTGBM_SPACE = [
    param('num_leaves', hp.uniformint, 2, 500, int),
    param('max_depth', hp.uniformint, 2, 50, int),
    param('learning_rate', hp.loguniform, -6.0, 0.0, float),
    param('n_estimators', hp.uniformint, 2, 500, int),
    param('min_data_in_leaf', hp.uniformint, 2, 500, int),
    param('lambda_l1', hp.loguniform, -6.0, 0.0, float),
    param('lambda_l2', hp.loguniform, -6.0, 0.0, float),
    param('linear_lambda', hp.loguniform, -6.0, 0.0, float),
    param('bagging_fraction', hp.uniform, 0.2, 1.0, float),
    param('bagging_freq', hp.uniformint, 0, 10, int),
    param('feature_fraction', hp.uniform, 0.2, 1.0, float),
    param('sigmoid', hp.uniform, 0.0, 5.0, float),
    param('verbosity', -1, None, None, int),
    param('objective', 'binary', None, None, str),
    param('eval_metric', 'binary_logloss', None, None, str),
]
