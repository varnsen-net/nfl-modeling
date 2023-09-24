"""Helper functions for evaluating models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss,
                             log_loss,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             confusion_matrix)


def append_array_to_scores(scores, metric_array, name):
    """Append an array of scores to a scores dictionary. Elements of the
    array are appended as key:value pairs.
    
    :param dict scores: scoring dict
    :param np.array metric_array: array of scores
    :param str name: name of metric
    :return: dictionary of scores
    :rtype: dict
    """
    for i, score in enumerate(metric_array):
        scores[f'{name}_{i}'] = score
    return scores


def custom_scorer(pipeline, X, y):
    """Score model using a variety of metrics.
    
    :param sklearn.pipeline.Pipeline pipeline: pipeline to evaluate
    :param pd.DataFrame X: features
    :param pd.Series y: target
    :return: evaluation metrics
    :rtype: dict
    """
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=7,
                                             strategy='quantile')
    scores = {'neg_brier_score': -brier_score_loss(y, y_pred_proba),
              'neg_log_loss': -log_loss(y, y_pred_proba),
              'f1': f1_score(y, y_pred),
              'precision': precision_score(y, y_pred),
              'recall': recall_score(y, y_pred),
              'roc_auc': roc_auc_score(y, y_pred_proba),
              'tn': tn,
              'fp': fp,
              'fn': fn,
              'tp': tp}
    scores = append_array_to_scores(scores, prob_true, 'prob_true_bin')
    scores = append_array_to_scores(scores, prob_pred, 'prob_pred_bin')
    return scores


def evaluate_model(pipeline, X, y, cv=5):
    """Evaluate model using a variety of metrics.
    
    :param sklearn.pipeline.Pipeline pipeline: pipeline to evaluate
    :param pd.DataFrame X: features
    :param pd.Series y: target
    :param int cv: number of cross-validation folds
    :return: evaluation metrics
    :rtype: pd.DataFrame
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=custom_scorer)
    scores = pd.DataFrame(scores).T
    scores.columns = [f'fold_{i+1}' for i in range(cv)]
    scores.index.name = 'metric'
    scores['mean'] = scores.mean(axis=1)
    scores['std'] = scores.std(axis=1)
    return scores
