"""Helper functions for producing model predictions."""

import numpy as np
import pandas as pd
from scipy.stats import mode


def voting_classifier(estimators, X, type):
    """Generate predictions for a set of fitted estimators.

    For soft voting, return the average of the predicted probabilities.
    For hard voting, return the most common prediction.
    
    :param list estimators: fitted estimators
    :param pd.DataFrame X: features
    :param str type: type of prediction. must be 'soft' or 'hard'
    :return: predictions
    :rtype: np.ndarray
    """
    if type == 'soft':
        return np.mean([est.predict_proba(X)[:,1] for est in estimators], axis=0)
    elif type == 'hard':
        return mode([est.predict(X) for est in estimators], axis=0)[0]
    else:
        return None
