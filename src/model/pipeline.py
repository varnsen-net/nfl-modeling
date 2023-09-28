"""Helper functions for building scikit-learn pipelines.

The base pipeline consists of a single preprocessor applicable to any training/test dataset. All other pipeline functions should build on top of that base and return a complete pipeline with an estimator."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.model.process import preprocessor, reduce_columns
from src.model.estimators import baseline_estimator


def base_pipeline(features_metadata):
    """Build a base pipeline classifier with a preprocessor. Full pipelines
    should be built by appending additional steps to this pipeline.

    :param dict features_metadata: feature metadata
    :return: base pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    kw_args = {'features': features_metadata}
    prep = FunctionTransformer(preprocessor, kw_args=kw_args)
    pipeline = make_pipeline(prep)
    return pipeline


def build_baseline_pipeline(features_metadata, model_params):
    """Build a baseline model pipeline on top of the base pipeline.
    
    :param dict features_metadata: feature metadata
    :param dict model_params: estimator parameters
    :return: baseline pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    pipeline = base_pipeline(features_metadata)
    feature_columns = ['away_pythagorean_expectation',
                       'home_pythagorean_expectation',
                       'away_rest',
                       'home_rest',
                       'away_travel_distance']
    kw_args = {'columns': feature_columns}
    column_reducer = FunctionTransformer(reduce_columns, kw_args=kw_args)
    pipeline.steps.append(('column_reducer', column_reducer))
    estimator = baseline_estimator(**model_params)
    pipeline.steps.append(('estimator', estimator))
    return pipeline
