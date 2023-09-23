import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def baseline_estimator(*args, **kwargs):
    return LogisticRegression(*args, **kwargs)
