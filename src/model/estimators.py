"""Build estimators to be used as the final step in a pipeline.

Baseline model: Logistic Regression
Working model: SWIFT
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


def baseline_estimator(*args, **kwargs):
    """Returns a calibrated logistic regression estimator.
    
    :param list args: positional arguments to pass to LogisticRegression
    :param dict kwargs: keyword arguments to pass to LogisticRegression
    :return: calibrated logistic regression estimator
    :rtype: sklearn.calibration.CalibratedClassifierCV
    """
    estimator = LogisticRegression(*args, **kwargs)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=5)
    return calibrated_estimator


def swift_estimator(*args, **kwargs):
    """Returns a calibrated SWIFT estimator.
    
    :param list args: positional arguments to pass to LogisticRegression
    :param dict kwargs: keyword arguments to pass to LogisticRegression
    :return: calibrated logistic regression estimator
    :rtype: sklearn.calibration.CalibratedClassifierCV
    """
    estimator = LGBMClassifier(*args, **kwargs)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=5)
    return calibrated_estimator
