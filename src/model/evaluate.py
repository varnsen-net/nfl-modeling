"""Helper functions for evaluating models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate


def evaluate_model(pipeline, X, y):
    """Evaluate model using cross-validation."""
    metrics = {'neg_brier_score': 'neg_brier_score',
               'f1': 'f1',
               'neg_log_loss': 'neg_log_loss',
               'precision': 'precision',
               'recall': 'recall',
               'roc_auc': 'roc_auc'}
    aggs = ['mean', 'std']
    scores = cross_validate(pipeline, X, y, scoring=metrics, cv=5)
    scores = pd.DataFrame(scores).agg(aggs).T
    scores.index.name = 'metric'
    return scores
