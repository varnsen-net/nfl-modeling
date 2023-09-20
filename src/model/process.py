"""Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html
"""

import numpy as np
import pandas as pd


def make_precision_map(features):
    """Make a precision map from a config file.
    
    :param dict features: features metadata from a config file
    :return: a mapping of column names to precision
    :rtype: dict
    """
    precision_map = {}
    for category in features:
        for feature in features[category]:
            precision_map[feature] = features[category][feature]['precision']
    return precision_map


def set_precision(X, features):
    """Set the precision for certain columns in a training or test set.

    :param pd.DataFrame X: a set of training or test data
    :param dict features: features metadata from a config file
    :return: a set of training or test data with precision set
    :rtype: pd.DataFrame
    """
    precision_map = make_precision_map(features)
    for col, precision in precision_map.items():
        feature = df.filter(regex=f"{col}$")
        df[feature.columns] = feature.round(precision)
    return df


