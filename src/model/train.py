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
from src.model.evaluate import evaluate_model, custom_cv
from src.plot.plot import make_and_save_plots


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
        features_metadata = config['features']

    train = pd.read_csv(f"{train_path}/train.csv", index_col=0)
    target = pd.read_csv(f"{train_path}/target.csv")
    X = preprocess(train, features_metadata)
    y = target['target'].values
    model_params = {'solver': 'liblinear'}
    baseline = build_baseline_pipeline(model_params)
    cv = custom_cv()
    scores = evaluate_model(baseline, X, y, cv)
    save_path = make_save_path(results_path)
    type = 'baseline'
    scores.to_csv(f"{save_path}/{type}_scores.csv")
    make_and_save_plots(scores, type, save_path)
    best_params = hyperoptimize(X, y, cv, LIGHTGBM_SPACE, max_evals=50)
    print("Best params:", best_params)
    swift = build_swift_pipeline(best_params)
    type = 'swift'
    scores = evaluate_model(swift, X, y, cv)
    scores.to_csv(f"{save_path}/{type}_scores.csv")
    make_and_save_plots(scores, type, save_path)
