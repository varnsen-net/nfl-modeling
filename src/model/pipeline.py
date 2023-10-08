"""Helper functions for building scikit-learn pipelines.

The base pipeline consists of a single preprocessor applicable to any training/test dataset. All other pipeline functions should build on top of that base and return a complete pipeline with an estimator."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.model.process import reduce_columns
from src.model.estimators import baseline_estimator, swift_estimator


def build_baseline_pipeline(model_params={}):
    """Build a baseline model pipeline.
    
    :param dict model_params: estimator parameters
    :return: baseline pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    feature_columns = ['obj_pythagorean_expectation',
                       'adv_pythagorean_expectation',
                       'obj_rest',
                       'adv_rest',
                       'obj_travel_distance',
                       'adv_travel_distance',
                       'obj_team_is_home']
    kw_args = {'columns': feature_columns}
    column_reducer = FunctionTransformer(reduce_columns, kw_args=kw_args)
    estimator = baseline_estimator(**model_params)
    pipeline = make_pipeline(column_reducer, estimator)
    return pipeline


def build_swift_pipeline(model_params={}):
    """Build a pipeline for a model that is actually good.
    
    :param dict model_params: estimator parameters
    :return: swift pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    transformers = [('onehot', OneHotEncoder(handle_unknown='ignore'), ['roof', 'surface'])]
    preprocessor = ColumnTransformer(transformers=transformers,
                                     remainder='passthrough')
    estimator = swift_estimator(**model_params)
    pipeline = make_pipeline(preprocessor, estimator)
    return pipeline
