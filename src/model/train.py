"""Trains models and records metadata about each training run."""

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.utils import collect_setup_args
from src.model.process import preprocessor, reduce_columns
from src.model.estimators import baseline_estimator
from src.model.pipeline import calibrated_classifier_pipeline


def build_baseline_model(preprocessor, features_metadata, model_params=None):
    """Builds a calibrated baseline classifier for training.
    
    :param callable preprocessor: preprocessor for pipeline
    :param dict features_metadata: features metadata
    :param dict model_params: estimator parameters
    :return: complete pipeline ready for fitting
    :rtype: sklearn.pipeline.Pipeline
    """
    estimator = baseline_estimator(**model_params)
    feature_columns = ['away_pythagorean_expectation',
                       'home_pythagorean_expectation',
                       'away_rest',
                       'home_rest',
                       'away_travel_distance']
    kw_args = {'columns': feature_columns}
    column_reducer = FunctionTransformer(reduce_columns, kw_args=kw_args)
    proc_steps = [('column_reducer', column_reducer)]
    pipeline = calibrated_classifier_pipeline(estimator, preprocessor,
                                              features_metadata, proc_steps)
    return pipeline


if __name__ == "__main__":
    args = collect_setup_args()

    config_path = args.c
    features_path = args.f
    train_path = args.tr
    test_path = args.te


    with open(config_path, 'r') as f:
        config = json.load(f)
        features_metadata = config['features']
    train = pd.read_csv(f"{train_path}/train.csv")
    target = pd.read_csv(f"{train_path}/target.csv")
    model_params = {'solver': 'liblinear'}
    baseline = build_baseline_model(preprocessor, features_metadata,
                                    model_params)
    baseline.fit(train, target['target'])
