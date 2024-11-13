import os
import pathlib


PROJ_ROOT = pathlib.Path(__file__).parents[2].absolute()
"""*pathlib.Path*: Root directory of the project."""
PATHS = {
    'raw_games': PROJ_ROOT / 'data' / 'raw' / 'games.csv',
    'raw_plays': PROJ_ROOT / 'data' / 'raw' / 'plays',
    'features': PROJ_ROOT / 'data' / 'features',
    'train': PROJ_ROOT / 'data' / 'train',
    'test': PROJ_ROOT / 'data' / 'test',
    'results': PROJ_ROOT / 'data' / 'results',
    'prediction': PROJ_ROOT / 'data' / 'prediction',
    'features_src': PROJ_ROOT / 'src' / 'data' / 'features',
    'city_coordinates': PROJ_ROOT / 'data' / 'ancillary' / 'city-coordinates.csv',
    'expected_values': PROJ_ROOT / 'data' / 'ancillary' / 'expected-values.csv',
}
"""*dict*: Paths to data files and directories."""
RAW_DATA_URLS = {
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "plays": "https://github.com/nflverse/nflverse-data/releases/download/pbp",
}
"""*dict*: URLs to raw data files."""


CURRENT_SEASON = 2024
"""*int*: Current NFL season. Used to fetch current season of play-by-play data."""
CURRENT_WEEK = 11
"""*int*: Current NFL week. Used to fetch current week of play-by-play data."""
TRAINING = {
    "holdout_year_start": 2021,
    "games_cols": ["game_id", "season", "week", "away_rest", "home_rest",
                   "away_team", "home_team"]
}
"""*dict*: Parameters for training data."""
SCORING_METRIC = 'neg_brier_score'
"""*str*: Scoring metric for hyperparameter tuning."""
CV_TRAIN_SIZE = 10
"""*int*: Number of consecutive seasons to use for training in grouped time-series cross validation."""
CV_TEST_SIZE = 3
"""*int*: Number of consecutive seasons to use for testing in grouped time-series cross validation."""
CV_SHIFT_SIZE = 2
"""*int*: Number of seasons to shift for each CV fold."""
MAX_EVALS = 20
"""*int*: Maximum number of hyperparameter tuning iterations."""
EARLY_STOP_N = 20
"""*int*: Number of iterations without improvement to stop hyperparameter tuning."""
DEFAULT_PARAM_PREFIX = 'calibratedclassifiercv__estimator__'
"""*str*: Prefix for default hyperparameters."""


LOOKBACK_WINDOW = 3
"""*int*: Number of weeks to look back for features that only consider recent performance."""


FEATURE_PRECISIONS = {
    "away_lon_delta": 2,
    "away_travel_distance": 0,
}
"""*dict*: Number of decimal places to round features to."""
