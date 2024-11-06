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
from src.data.features.game_stats import build_game_features
from src.data.features.drive_stats import build_drive_features
from src.data.features.series_stats import build_series_features
from src.data.features.play_stats import build_play_features
from src.data.train.train import build_train
from src.data.train.target import build_target

from src.config.config import (TRAINING,
                               CURRENT_SEASON,
                               CURRENT_WEEK,
                               RAW_DATA_URLS,
                               PATHS)


def preprocess_plays(raw_plays):
    """Preprocess raw play-by-play data.

    :param pd.DataFrame raw_plays: Raw play-by-play data.
    :return: Preprocessed play-by-play data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .query('posteam != "" and posteam != "None"')
            .query('half_seconds_remaining > 120')
            .query('score_differential.abs() <= 16'))


def split_data(df, holdout_year_start):
    """Divide a full set of features into training and holdout data.

    :param df: Full set of features data.
    :type df: pd.DataFrame of shape (n_samples, n_features)
    :param int holdout_year_start: Starting season of holdout data (inclusive).
    :return: Training and holdout data
    :rtype: (pd.DataFrame, pd.DataFrame) 
    """
    train = df.loc[df["season"] < holdout_year_start]
    holdout = df.loc[df["season"] >= holdout_year_start]
    return train, holdout


def build_train_and_test_data(train_path, test_path, games_cols, raw_games_path,
                              features_path, holdout_year_start):
    """Build training and testing data from raw data and save to local paths.
    
    :param str train_path: Path to save training data.
    :param str test_path: Path to save testing data.
    :param list games_cols: Columns to keep from raw games data.
    :param str raw_games_path: Path to raw games data.
    :param str features_path: Path to features data.
    :param int holdout_year_start: Starting season of holdout data (inclusive).
    :return: None
    :rtype: None
    """
    full_train = build_train(games_cols, raw_games_path, features_path)
    full_target = build_target(raw_games_path, full_train)
    train, train_holdout = split_data(full_train, holdout_year_start)
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
    holdout_year_start = TRAINING['holdout_year_start']
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

    game_features = pd.DataFrame()
    drive_features = pd.DataFrame()
    series_features = pd.DataFrame()
    play_features = pd.DataFrame()
    for season in list(range(2001, CURRENT_SEASON + 1)):
        print(f"Building features for season {season}")
        raw_plays = raw_plays_path / f"play_by_play_{season}.parquet"
        raw_plays = pd.read_parquet(raw_plays)

        print("Preprocessing...")
        processed_plays = preprocess_plays(raw_plays)

        print("Building game features...")
        features = build_game_features(processed_plays, season)
        game_features = pd.concat([game_features, features])

        print('Building drive features...')
        features = build_drive_features(processed_plays, season)
        drive_features = pd.concat([drive_features, features])

        print('Building series features...')
        features = build_series_features(processed_plays, season)
        series_features = pd.concat([series_features, features])

        print('Building play features...')
        features = build_play_features(processed_plays, season)
        play_features = pd.concat([play_features, features])

    game_features.to_csv(features_path / 'game_features.csv')
    drive_features.to_csv(features_path / 'drive_features.csv')
    series_features.to_csv(features_path / 'series_features.csv')
    play_features.to_csv(features_path / 'play_features.csv')

    print('Building training and test data...')
    build_train_and_test_data(train_path, test_path, games_cols, raw_games_path,
                              features_path, holdout_year_start)
