"""Trains models and records metadata about each training run."""

import json
import datetime
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.utils import collect_setup_args
from src.model.process import preprocessor, reduce_columns
from src.model.estimators import baseline_estimator
from src.model.pipeline import calibrated_classifier_pipeline
from src.model.evaluate import evaluate_model


def create_datetime_id():
    """Creates a unique identifier for the current datetime.
    
    :return: unique identifier
    :rtype: str
    """
    now = datetime.datetime.now()
    dt_id = now.strftime("%Y%m%d%H%M%S")
    return dt_id


def save_model_and_scores(model, scores, results_path, type):
    """Saves a model to a unique directory.
    
    :param sklearn.base.BaseEstimator model: model to save
    :param pd.DataFrame scores: scores for model
    :param str results_path: path to training results directory
    :param str type: type of model
    :return: None
    :rtype: None
    """
    dt_id = create_datetime_id()
    save_path = f"{results_path}/{dt_id}"
    os.makedirs(save_path)
    joblib.dump(model, f"{save_path}/{type}_model.joblib")
    scores.to_csv(f"{save_path}/{type}_scores.csv")
    return 


def build_baseline_model(preprocessor, features_metadata, model_params={}):
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
    results_path = args.r

    with open(config_path, 'r') as f:
        config = json.load(f)
        features_metadata = config['features']

    train = pd.read_csv(f"{train_path}/train.csv")
    target = pd.read_csv(f"{train_path}/target.csv")
    model_params = {'solver': 'liblinear'}
    baseline = build_baseline_model(preprocessor, features_metadata,
                                    model_params)
    scores = evaluate_model(baseline, train, target['target'])
    baseline.fit(train, target['target'])
    save_model_and_scores(baseline, scores, results_path, 'baseline')
