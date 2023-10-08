"""Train and evaluate models."""

import json
import datetime
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GroupKFold
from hyperopt import tpe, hp, fmin

from src.utils import collect_setup_args
from src.model.process import preprocess
from src.model.pipeline import build_baseline_pipeline, build_swift_pipeline
from src.model.hyperoptimize import hyperoptimize, LIGHTGBM_SPACE
from src.model.evaluate import custom_cv, evaluate_model, compile_scores
from src.model.predict import voting_classifier
from src.plot.plot import make_and_save_plots, plot_test_calibration


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
    args = collect_setup_args()
    config_path = args.c
    features_path = args.f
    train_path = args.tr
    test_path = args.te
    results_path = args.r

    with open(config_path, 'r') as f:
        config = json.load(f)
        feature_precisions = config['feature_precisions']

    train = pd.read_csv(f"{train_path}/train.csv", index_col=0)
    target = pd.read_csv(f"{train_path}/target.csv", index_col=0)
    X = preprocess(train, feature_precisions)
    y = target['target']
    cv = custom_cv()

    # evaluate baseline model
    baseline = build_baseline_pipeline({'solver': 'liblinear'})
    scores, _ = evaluate_model(baseline, X, y, cv)
    save_path = make_save_path(results_path)
    name = 'baseline'
    scores.to_csv(f"{save_path}/{name}_scores.csv")
    make_and_save_plots(scores, name, save_path)

    # evaluate swift
    best_params = hyperoptimize(X, y, cv, LIGHTGBM_SPACE, max_evals=50)
    swift = build_swift_pipeline(best_params)
    scores, estimators = evaluate_model(swift, X, y, cv)
    name = 'swift'
    scores.to_csv(f"{save_path}/{name}_scores.csv")
    make_and_save_plots(scores, name, save_path)

    # evaluate swift on holdout data
    print(f"Evaluating {name} on holdout data...")
    test = pd.read_csv(f"{test_path}/test.csv", index_col=0)
    target = pd.read_csv(f"{test_path}/target.csv", index_col=0)
    X_test = preprocess(test, feature_precisions)
    y_test = target['target']
    y_pred = voting_classifier(estimators, X_test, 'hard')
    y_pred_proba = voting_classifier(estimators, X_test, 'soft')
    scores = compile_scores(y_test, y_pred, y_pred_proba)
    plot_test_calibration(scores, name, save_path)
    print("Best params:", best_params)

