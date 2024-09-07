"""Train and evaluate models."""

import json
import datetime
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GroupKFold

from src.model.process import preprocess, transform_home_away_structure
from src.model.estimators import build_baseline_pipeline, build_swift_pipeline
from src.model.hyperoptimize import hyperoptimize
from src.model.evaluate import custom_cv, evaluate_model, compile_scores
from src.model.predict import voting_classifier
from src.plot.plot import make_and_save_plots, plot_test_calibration

from src.config.config import (PATHS,
                               FEATURE_PRECISIONS,
                               CV_TRAIN_SIZE,
                               CV_TEST_SIZE,
                               SCORING_METRIC,
                               MAX_EVALS,
                               EARLY_STOP_N)
from src.config.spaces import BASELINE_PARAMS, LIGHTGBM_SPACE


def create_datetime_id():
    """Creates a unique identifier for the current datetime.
    
    :return: unique identifier
    :rtype: str
    """
    now = datetime.datetime.now()
    dt_id = now.strftime("%Y%m%d%H%M%S")
    return dt_id


def make_save_path(results_path):
    """Creates a directory for saving model results.
    
    :param str results_path: path to training results directory
    :return: path to directory for model results
    :rtype: str
    """
    dt_id = create_datetime_id()
    save_path = f"{results_path}/{dt_id}"
    os.makedirs(save_path)
    return save_path


if __name__ == "__main__":
    features_path = PATHS['features']
    train_path = PATHS['train']
    test_path = PATHS['test']
    results_path = PATHS['results']
    save_path = make_save_path(results_path)

    X_train = pd.read_csv(f"{train_path}/train.csv", index_col=0)
    y_train = pd.read_csv(f"{train_path}/target.csv", index_col=0)
    X_train, y_train = transform_home_away_structure(X_train, y_train)
    X_train = preprocess(X_train, FEATURE_PRECISIONS)
    y_train = y_train['target']

    X_test = pd.read_csv(f"{test_path}/test.csv", index_col=0)
    y_test = pd.read_csv(f"{test_path}/target.csv", index_col=0)
    X_test, y_test = transform_home_away_structure(X_test, y_test)
    X_test = preprocess(X_test, FEATURE_PRECISIONS)
    y_test = y_test['target']

    cv = custom_cv(CV_TRAIN_SIZE, CV_TEST_SIZE)

    # evaluate baseline model
    name = 'baseline'
    print(f"Evaluating {name} on training and holdout data...")
    baseline = build_baseline_pipeline(BASELINE_PARAMS)
    scores, bl_estimators = evaluate_model(baseline, X_train, y_train, cv)
    scores.to_csv(f"{save_path}/{name}_scores.csv")
    make_and_save_plots(scores, name, save_path)
    y_pred = voting_classifier(bl_estimators, X_test, 'hard')
    y_pred_proba = voting_classifier(bl_estimators, X_test, 'soft')
    scores = compile_scores(y_test, y_pred, y_pred_proba)
    plot_test_calibration(scores, name, save_path)

    # evaluate swift
    name = 'swift'
    print(f"Evaluating {name} on training and holdout data...")
    swift = build_swift_pipeline()
    best_params = hyperoptimize(swift, X_train, y_train, cv,
                                scoring=SCORING_METRIC,
                                space=LIGHTGBM_SPACE,
                                max_evals=MAX_EVALS,
                                early_stop_n=EARLY_STOP_N)
    print(f"Best params: {best_params}")
    swift.set_params(**best_params)
    scores, sw_estimators = evaluate_model(swift, X_train, y_train, cv)
    scores.to_csv(f"{save_path}/{name}_scores.csv")
    make_and_save_plots(scores, name, save_path)
    y_pred = voting_classifier(sw_estimators, X_test, 'hard')
    y_pred_proba = voting_classifier(sw_estimators, X_test, 'soft')
    scores = compile_scores(y_test, y_pred, y_pred_proba)
    plot_test_calibration(scores, name, save_path)

    # train models on all data and save
    print(f"Training {name} on all data...")
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    baseline.fit(X_full, y_full)
    joblib.dump(baseline, f"{save_path}/baseline_model.pkl")
    swift.fit(X_full, y_full)
    joblib.dump(swift, f"{save_path}/swift_model.pkl")
