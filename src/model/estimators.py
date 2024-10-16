"""Build estimators to be used as the final step in a pipeline.

Baseline model: Logistic Regression
Working model: SWIFT
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import NuSVC
from sklearn.calibration import CalibratedClassifierCV

from src.model.process import reduce_columns, drop_columns


def build_baseline_pipeline(model_params={}):
    """Build a baseline model pipeline.
    
    :param dict model_params: estimator parameters
    :return: baseline pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    feature_columns = ['obj_rest',
                       'adv_rest',
                       'obj_travel_distance',
                       'adv_travel_distance']
    kw_args = {'columns': feature_columns}
    column_reducer = FunctionTransformer(reduce_columns, kw_args=kw_args)
    estimator = LogisticRegression(**model_params)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=3)
    pipeline = make_pipeline(column_reducer, StandardScaler(),
                             calibrated_estimator)
    return pipeline


def build_lgbm_pipeline(model_params={}):
    """Build an LGBM pipeline.
    
    :param dict model_params: estimator parameters
    :return: swift pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    cols_to_drop = ['season']
    kw_args = {'columns': cols_to_drop}
    column_reducer = FunctionTransformer(drop_columns, kw_args=kw_args)
    estimator = LGBMClassifier(**model_params)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=3)
    pipeline = make_pipeline(column_reducer, calibrated_estimator)
    return pipeline


def build_svc_pipeline(model_params={}):
    """Build an SVC pipeline.
    
    :param dict model_params: estimator parameters
    :return: swift pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    cols_to_drop = ['season']
    kw_args = {'columns': cols_to_drop}
    column_reducer = FunctionTransformer(drop_columns, kw_args=kw_args)
    estimator = NuSVC(**model_params)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=3)
    pipeline = make_pipeline(column_reducer, StandardScaler(),
                             calibrated_estimator)
    return pipeline
