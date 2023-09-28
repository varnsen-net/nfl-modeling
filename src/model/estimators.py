"""Build estimators to be used as the final step in a pipeline.

Baseline model: Logistic Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def baseline_estimator(*args, **kwargs):
    """Returns a calibrated logistic regression estimator."""
    estimator = LogisticRegression(*args, **kwargs)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=5)
    return calibrated_estimator
