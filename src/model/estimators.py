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
    """Returns a calibrated logistic regression estimator."""
    estimator = LogisticRegression(*args, **kwargs)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=5)
    return calibrated_estimator


def swift_estimator(*args, **kwargs):
    """Returns a calibrated SWIFT estimator."""
    estimator = LGBMClassifier(*args, **kwargs)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=5)
    return calibrated_estimator
