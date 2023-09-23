"""Helper functions for building scikit-learn pipelines."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV


def calibrated_classifier_pipeline(estimator, preprocessor, features_metadata,
                                   proc_steps=None):
    """Build a scikit-learn pipeline with a calibrated classifier.

    :param sklearn.base.BaseEstimator estimator: model for last pipeline step
    :param callable preprocessor: function to preprocess data
    :param list features_metadata: feature metadata
    :param list[tuple] proc_steps: additional pipeline steps
    :return: scikit-learn pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    kw_args = {'features': features_metadata}
    preprocessor = FunctionTransformer(preprocessor, kw_args=kw_args)
    pipeline = make_pipeline(preprocessor)
    if proc_steps:
        pipeline.steps.extend(proc_steps)
    calibrated_estimator = CalibratedClassifierCV(estimator, cv=5)
    pipeline.steps.append(('calibrated', calibrated_estimator))
    return pipeline
