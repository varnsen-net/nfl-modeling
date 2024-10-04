"""Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

The preprocess function is reserved for model-agnostic transformations that should be done before building the scikit pipelines.
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


def drop_columns(X, columns):
    """Drop a set of columns from a training or test set.

    :param pd.DataFrame X: a set of training or test data
    :param list columns: a list of columns to drop
    :return: a set of training or test data with columns dropped
    :rtype: pd.DataFrame
    """
    return X.drop(columns=columns,
                  errors='ignore')


def transform_home_away_structure(X, y):
    """Convert home/away structure to object/adversary structure.
    
    :param pd.DataFrame X: a set of training or test data
    :param pd.DataFrame y: a set of training or test labels
    :return: a set of training or test data with object/adversary structure
    :rtype: pd.DataFrame
    """
    home_obj_idxs = X[::2].index
    home_obj_games = X.loc[home_obj_idxs]
    home_obj_games.columns = (home_obj_games.columns
                              .str.replace("^home", "obj", regex=True)
                              .str.replace("^away", "adv", regex=True))
    home_obj_games['obj_team_is_home'] = 1
    home_obj_y = y.loc[home_obj_idxs]

    away_obj_games = X.drop(home_obj_idxs)
    away_obj_games.columns = (away_obj_games.columns
                              .str.replace("^away", "obj", regex=True)
                              .str.replace("^home", "adv", regex=True))
    away_obj_games['obj_team_is_home'] = 0
    away_obj_y = y.drop(home_obj_idxs)
    away_obj_y = 1 - away_obj_y

    X_transformed = pd.concat([home_obj_games, away_obj_games]).sort_index()
    y_transformed = pd.concat([home_obj_y, away_obj_y]).sort_index()

    return X_transformed, y_transformed
