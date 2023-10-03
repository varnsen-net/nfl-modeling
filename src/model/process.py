"""Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

The preprocess function is reserved for model-agnostic transformations that should be done at the start of a pipeline.
"""

import numpy as np
import pandas as pd


def set_precision(X, precision_map):
    """Set the precision for certain columns in a training or test set.

    :param pd.DataFrame X: a set of training or test data
    :param dict precision_map: mapping of column names and precision values
    :return: a set of training or test data with precision set
    :rtype: pd.DataFrame
    """
    for col, precision in precision_map.items():
        feature = X.filter(regex=f"{col}$")
        col_names = feature.columns
        X = X.drop(columns=col_names)
        X[col_names] = feature.round(precision)
    return X


def preprocess(X, features):
    """Apply preprocessing transformations.

    :param pd.DataFrame X: a set of training or test data
    :param dict features: features metadata from a config file
    :return: preprocessed training or test data
    :rtype: pd.DataFrame
    """
    X = set_precision(X, features)
    return X


def reduce_columns(X, columns):
    """Reduce a set of training or test data to a subset of columns.

    :param pd.DataFrame X: a set of training or test data
    :param list columns: a list of columns to keep
    :return: a set of training or test data with a subset of columns
    :rtype: pd.DataFrame
    """
    return X[columns]
