"""Build the full set of training and testing data from scratch.

Receives a set of paths from setup.sh, then fetches raw data, builds features, and assembles training data. Then it splits holdout data by year and writes to disk.
"""

import os
import pathlib

import numpy as np
import pandas as pd

from src.data.raw.games import refresh_games_data
from src.data.raw.plays import refresh_plays_data
from src.data.features.travel import build_travel_features
from src.data.features.points import build_points_features
from src.data.features.team_stats import build_team_efficiency_features
from src.data.train.train import build_train
from src.data.train.target import build_target

from src.config.config import (TRAINING,
                               CURRENT_SEASON,
                               CURRENT_WEEK,
                               RAW_DATA_URLS,
                               PATHS)


def split_data(df, holdout_year):
    """Divide a full set of features into training and holdout data.

    :param df: Full set of features data.
    :type df: pd.DataFrame of shape (n_samples, n_features)
    :param int holdout_year: Starting season of holdout data (inclusive).
    :return: Training and holdout data
    :rtype: (pd.DataFrame, pd.DataFrame) 
    """
    train = df.loc[df["season"] < holdout_year]
    holdout = df.loc[df["season"] >= holdout_year]
    return train, holdout


def build_train_and_test_data(train_path, test_path, games_cols, raw_games_path,
                              features_path, holdout_year):
    """Build training and testing data from raw data and save to local paths.
    
    :param str train_path: Path to save training data.
    :param str test_path: Path to save testing data.
    :param list games_cols: Columns to keep from raw games data.
    :param str raw_games_path: Path to raw games data.
    :param str features_path: Path to features data.
    :param int holdout_year: Starting season of holdout data (inclusive).
    :return: None
    :rtype: None
    """
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
    raw_plays_path = PATHS['raw_plays']
    city_coords_path = PATHS['city_coordinates']
    features_path = PATHS['features']
    train_path = PATHS['train']
    test_path = PATHS['test']
    games_cols = TRAINING['games_cols']
    holdout_year = TRAINING['holdout_year']
    games_url = RAW_DATA_URLS['games']
    plays_url = RAW_DATA_URLS['plays']

    os.makedirs(features_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print('Refreshing raw games data...')
    refresh_games_data(games_url, raw_games_path)

    print('Refreshing raw play-by-play data...')
    refresh_plays_data(CURRENT_SEASON, plays_url, raw_plays_path)

    # TODO throw these calls into a dictionary or something
    print('Building travel features...')
    travel_features = build_travel_features(raw_games_path, city_coords_path)
    output_path = features_path / 'travel.csv'
    travel_features.to_csv(output_path)

    print('Building points features...')
    points_features = build_points_features(raw_games_path, window=None)
    output_path = features_path / 'points.csv'
    points_features.to_csv(output_path)

    print('Building team efficiency features...')
    team_efficiency_features = build_team_efficiency_features(raw_plays_path)
    output_path = features_path / 'team_efficiency.csv'
    team_efficiency_features.to_csv(output_path)

    print('Building training and test data...')
    build_train_and_test_data(train_path, test_path, games_cols, raw_games_path,
                              features_path, holdout_year)
