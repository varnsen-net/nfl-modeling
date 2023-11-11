"""Build the full set of training and testing data from scratch.

Receives a set of paths from setup.sh, then fetches raw data, builds features, and assembles training data. Then it splits holdout data by year and writes to disk.
"""

import os
import pathlib

import numpy as np
import pandas as pd

from src.utils import collect_setup_args
from src.data.raw.games import refresh_games_data
from src.data.raw.weather import refresh_weather_data
from src.data.raw.plays import refresh_plays_data
from src.data.features.team_stats import build_team_stats_features
from src.data.features.travel import build_travel_features
from src.data.features.weather import build_weather_features
from src.data.train.train import build_train
from src.data.train.target import build_target

from src.config.config import (WEATHER_METADATA, TRAINING, CURRENT_SEASON,
                               CURRENT_WEEK, RAW_DATA_URLS, PATHS)


def split_data(df, holdout_year):
    """Divide a full set of features into training and holdout data.

    :param df: *pd.DataFrame of shape (n_samples, n_features)*
        Full set of features data.
    :param holdout_year: *int*
        Starting season of holdout data (inclusive).
    :return: *(pd.DataFrame, pd.DataFrame)*
        Training and holdout data
    """
    train = df.loc[df["season"] < holdout_year]
    holdout = df.loc[df["season"] >= holdout_year]
    return train, holdout


def refresh_raw_data(games_url, raw_games_path, weather_metadata,
                     raw_weather_path, city_coords_path, current_season,
                     plays_url, raw_plays_path):
    """Fetch raw data from web sources and save to local paths.

    :param games_url: *str*
        URL of raw games repository.
    :param raw_games_path: *str*
        Path to save raw games data.
    :param weather_metadata: *dict*
        Metadata for weather data, e.g. default values and agg functions to
        apply to game window. See: src.config.config for details.
    :param raw_weather_path: *str*
        Path to save raw weather data.
    :param city_coords_path: *str*
        Path to local city coordinates data. Required to fetch raw weather.
    :param current_season: *int*
        Current season, e.g. 2023.
    :param plays_url: *str*
        URL of raw plays repository.
    :param raw_plays_path: *str*
        Path to save raw plays data.
    :return: *None*
    """
    print('Refreshing raw games data...')
    refresh_games_data(games_url, raw_games_path)
    print('Refreshing raw weather data...')
    refresh_weather_data(weather_metadata, raw_games_path, raw_weather_path,
                         city_coords_path, batch_size=1000)
    print('Refreshing raw play-by-play data...')
    refresh_plays_data(current_season, plays_url, raw_plays_path)
    return


def build_features(features_path, weather_metadata,
                       raw_games_path, raw_weather_path, raw_plays_path,
                       city_coords_path):
    """Build features from raw data and save to local paths.
    
    :param features_path: *str*
        Path to save features data.
    :param weather_metadata: *dict*
        Metadata for weather data, e.g. default values and agg functions to
        apply to game window. See: src.config.config for details.
    :param raw_games_path: *str*
        Path to raw games data.
    :param raw_weather_path: *str*
        Path to raw weather data.
    :param raw_plays_path: *str*
        Path to raw play-by-play data.
    :param city_coords_path: *str*
        Path to local city coordinates data. Required to build travel distances.
    :return: *None*
    """
    features_path = pathlib.Path(features_path)
    print('Building team stats features...')
    output_dir = features_path / 'team_stats'
    os.makedirs(output_dir, exist_ok=True)
    build_team_stats_features(raw_games_path, raw_plays_path, output_dir)

    print('Building travel features...')
    output_dir = features_path / 'travel'
    os.makedirs(output_dir, exist_ok=True)
    build_travel_features(raw_games_path, city_coords_path, output_dir)

    print('Building weather features...')
    output_dir = features_path / 'weather'
    os.makedirs(output_dir, exist_ok=True)
    build_weather_features(weather_metadata, raw_games_path, raw_weather_path,
                           output_dir)
    return


def build_train_and_test_data(train_path, test_path, games_cols, raw_games_path,
                              features_path, holdout_year):
    """Build training and testing data from raw data and save to local paths.
    
    :param train_path: *str*
        Path to save training data.
    :param test_path: *str*
        Path to save testing data.
    :param games_cols: *list*
        List of columns to keep from raw games data.
    :param raw_games_path: *str*
        Path to raw games data.
    :param features_path: *str*
        Path to features data.
    :param holdout_year: *int*
        Starting season of holdout data (inclusive).
    :return: *None*
    """
    print('Building training and test data...')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    full_train = build_train(games_cols, raw_games_path, features_path)
    full_target = build_target(raw_games_path, full_train)
    train, train_holdout = split_data(full_train, holdout_year)
    target = full_target.loc[full_target["game_id"].isin(train["game_id"])]
    target_holdout = full_target.loc[full_target["game_id"].isin(train_holdout["game_id"])]
    train.to_csv(f'{train_path}/train.csv', index=False)
    target.to_csv(f'{train_path}/target.csv', index=False)
    train_holdout.to_csv(f'{test_path}/test.csv', index=False)
    target_holdout.to_csv(f'{test_path}/target.csv', index=False)
    return


if __name__ == '__main__':
    raw_games_path = PATHS['raw_games']
    raw_weather_path = PATHS['raw_weather']
    raw_plays_path = PATHS['raw_plays']
    city_coords_path = PATHS['city_coordinates']
    features_path = PATHS['features']
    train_path = PATHS['train']
    test_path = PATHS['test']
    games_cols = TRAINING['games_cols']
    holdout_year = TRAINING['holdout_year']
    games_url = RAW_DATA_URLS['games']
    plays_url = RAW_DATA_URLS['plays']


    # Refresh raw data
    refresh_raw_data(games_url, raw_games_path, WEATHER_METADATA,
                     raw_weather_path, city_coords_path, CURRENT_SEASON,
                     plays_url, raw_plays_path)


    # Build features
    build_features(features_path, WEATHER_METADATA,
                   raw_games_path, raw_weather_path, raw_plays_path,
                   city_coords_path)


    # Build training and test data
    build_train_and_test_data(train_path, test_path, games_cols, raw_games_path,
                              features_path, holdout_year)
