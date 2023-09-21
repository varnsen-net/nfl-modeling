"""Helper functions for building scikit-learn pipelines."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.model.process import preprocess


def build_pipeline(features):
    """Build a base scikit-learn pipeline.

    After calling this function in a training script, simply use 
    pipeline.steps.append() to add to the pipeline.
    
    :param dict features: features metadata
    :return: scikit-learn pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    preproc_kw_args = {'features': features}
    preprocessor = FunctionTransformer(preprocess, kw_args=preproc_kw_args)
    pipeline = make_pipeline(preprocessor)
    return pipeline
