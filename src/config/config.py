import os
import pathlib


PROJ_ROOT = pathlib.Path(__file__).parents[2].absolute()
"""*pathlib.Path*: Root directory of the project."""
PATHS = {
    'raw_games': PROJ_ROOT / 'data' / 'raw' / 'games.csv',
    'raw_plays': PROJ_ROOT / 'data' / 'raw' / 'plays',
    'raw_weather': PROJ_ROOT / 'data' / 'raw' / 'weather.csv',
    'city_coordinates': PROJ_ROOT / 'data' / 'ancillary' / 'city-coordinates.csv',
    'features': PROJ_ROOT / 'data' / 'features',
    'train': PROJ_ROOT / 'data' / 'train',
    'test': PROJ_ROOT / 'data' / 'test',
    'results': PROJ_ROOT / 'data' / 'results',
    'features_src': PROJ_ROOT / 'src' / 'data' / 'features',
}
"""*dict*: Paths to data files and directories."""
RAW_DATA_URLS = {
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "weather_hist": "https://archive-api.open-meteo.com/v1/archive",
    "weather_forecast": "https://api.open-meteo.com/v1/forecast",
    "plays": "https://github.com/nflverse/nflverse-data/releases/download/pbp",
}
"""*dict*: URLs to raw data files."""


CURRENT_SEASON = 2023
"""*int*: Current NFL season. Used to fetch current season of play-by-play data."""
CURRENT_WEEK = 12
"""*int*: Current NFL week. Used to fetch current week of play-by-play data."""
TRAINING = {
    "holdout_year": 2020,
    "games_cols": ["game_id", "season", "week", "away_rest", "home_rest",
                   "div_game", "roof", "surface", "away_team", "home_team"]
}
"""*dict*: Parameters for training data."""
SCORING_METRIC = 'neg_brier_score'
"""*str*: Scoring metric for hyperparameter tuning."""
CV_TRAIN_SIZE = 5
"""*int*: Number of consecutive seasons to use for training in grouped time-series cross validation."""
CV_TEST_SIZE = 3
"""*int*: Number of consecutive seasons to use for testing in grouped time-series cross validation."""
MAX_EVALS = 100
"""*int*: Maximum number of hyperparameter tuning iterations."""
EARLY_STOP_N = 10
"""*int*: Number of iterations without improvement to stop hyperparameter tuning."""
DEFAULT_PARAM_PREFIX = 'calibratedclassifiercv__estimator__'
"""*str*: Prefix for default hyperparameters."""


LOOKBACK_WINDOW = 3
"""*int*: Number of weeks to look back for features that only consider recent performance."""


WEATHER_METADATA = {
    "temperature": {
        "api_name": "temperature_2m",
        "agg_func": "mean",
        "default": 21.0
    },
    "apparent_temperature": {
        "api_name": "apparent_temperature",
        "agg_func": "mean",
        "default": 21.0
    },
    "humidity": {
        "api_name": "relativehumidity_2m",
        "agg_func": "mean",
        "default": 50.0
    },
    "dewpoint": {
        "api_name": "dewpoint_2m",
        "agg_func": "mean",
        "default": 12.0
    },
    "pressure": {
        "api_name": "surface_pressure",
        "agg_func": "mean",
        "default": None
    },
    "wind_speed": {
        "api_name": "windspeed_10m",
        "agg_func": "mean",
        "default": 0.0
    },
    "cloud_cover": {
        "api_name": "cloudcover",
        "agg_func": "mean",
        "default": 100.0
    },
    "rain": {
        "api_name": "rain",
        "agg_func": "sum",
        "default": 0.0
    },
    "snow": {
        "api_name": "snowfall",
        "agg_func": "sum",
        "default": 0.0
    },
}
"""*dict*: Metadata for weather features."""
FEATURE_PRECISIONS = {
    "temperature_2m": 0,
    "apparent_temperature": 0,
    "relativehumidity_2m": 1,
    "dewpoint_2m": 1,
    "surface_pressure": 0,
    "windspeed_10m": 1,
    "cloudcover": 0,
    "rain": 1,
    "snowfall": 1,
    "away_lon_delta": 2,
    "away_travel_distance": 0,
    "pythagorean_expectation": 3,
}
"""*dict*: Number of decimal places to round features to."""


PASSING_AGGS = {'week': ['max'],
                'pass': 'sum',
                'yards_gained': 'mean',
                'complete_pass': 'mean',
                'interception': 'mean',
                'fumble': 'mean',
                'sack': 'mean'}
"""*dict*: Functions for aggregating passing features from play-by-play data."""
RUSHING_AGGS = {'week': ['max'],
                'rush': 'sum',
                'yards_gained': 'mean',
                'fumble': 'mean'}
"""*dict*: Functions for aggregating rushing features from play-by-play data."""
