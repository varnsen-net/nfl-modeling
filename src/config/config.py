"""General configuration file. Stores paths, model training parameters, and other constants.

:PROJ_ROOT (pathlib.Path): Project root directory.
:PATHS (dict): Paths to data files and directories.
:RAW_DATA_URLS (dict): URLs to raw data files.
:CURRENT_SEASON (int): Current NFL season. Used to fetch current season of play-by-play data.
:CURRENT_WEEK (int): Current NFL week. Used to fetch current week of play-by-play data.
:TRAINING (dict): Parameters for training data.
:SCORING_METRIC (str): Scoring metric for hyperparameter tuning.
:CV_TRAIN_SIZE (int): Number of consecutive seasons to use for training in grouped time-series cross validation.
:CV_TEST_SIZE (int): Number of consecutive seasons to use for testing in grouped time-series cross validation.
:CV_SHIFT_SIZE (int): Number of seasons to shift for each CV fold.
:MAX_EVALS (int): Maximum number of hyperparameter tuning iterations.
:EARLY_STOP_N (int): Number of iterations without improvement to stop hyperparameter tuning.
:DEFAULT_PARAM_PREFIX (str): Prefix for default hyperparameters.
:FEATURE_PRECISIONS (dict): Number of decimal places to round features to.
"""

import os
import pathlib


PROJ_ROOT = pathlib.Path(__file__).parents[2].absolute()
PATHS = {
    'raw_games': PROJ_ROOT / 'data' / 'raw' / 'games.csv',
    'raw_plays': PROJ_ROOT / 'data' / 'raw' / 'plays',
    'features': PROJ_ROOT / 'data' / 'features',
    'train': PROJ_ROOT / 'data' / 'train',
    'results': PROJ_ROOT / 'data' / 'results',
    'prediction': PROJ_ROOT / 'data' / 'prediction',
    'features_src': PROJ_ROOT / 'src' / 'data' / 'features',
    'city_coordinates': PROJ_ROOT / 'data' / 'ancillary' / 'city-coordinates.csv',
    'expected_values': PROJ_ROOT / 'data' / 'ancillary' / 'expected-values.csv',
}
RAW_DATA_URLS = {
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "plays": "https://github.com/nflverse/nflverse-data/releases/download/pbp",
}


CURRENT_SEASON = 2024
CURRENT_WEEK = 99
TRAINING = {
    "min_year": 2001,
    "holdout_year_start": 2021,
    "games_cols": ["game_id", "season", "week", "away_team", "home_team",
                   "away_rest", "home_rest", "result"]
}
SCORING_METRIC = 'neg_brier_score'
CV_TRAIN_SIZE = 10
CV_TEST_SIZE = 3
CV_SHIFT_SIZE = 2
MAX_EVALS = 30
EARLY_STOP_N = 15
DEFAULT_PARAM_PREFIX = 'calibratedclassifiercv__estimator__'
FEATURE_PRECISIONS = {
    "away_lon_delta": 2,
    "away_travel_distance": 0,
}
