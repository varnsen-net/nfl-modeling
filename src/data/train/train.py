"""Helper functions for building the training data."""

import os

import numpy as np
import pandas as pd

from src.utils import (shift_week_number,
                       map_features_to_games,
                       walk_features_dir)


def preprocess_raw_games(games):
    """Reduce the games dataframe.
    
    :param pd.DataFrame games: Raw games dataframe.
    :return: Reduced games data.
    :rtype: pd.DataFrame
    """
    games = (games
             .loc[games['week'] > 3]
             .loc[~((games['week'] > 16) & (games['season'] < 2021))]
             .loc[~((games['week'] > 17) & (games['season'] >= 2021))]
             .loc[games['game_type'] == 'REG']
             .loc[games['location'] == 'Home']
             .drop(columns=['game_type', 'location'])
             .dropna(subset=['result'])
             .copy())
    return games


def merge_feature(train, features):
    """Merge a feature into the training data.
    
    :param pd.DataFrame train: Training data.
    :param pd.DataFrame features: Feature data.
    :return: Training data with feature merged in.
    :rtype: pd.DataFrame
    """
    first_col = features.columns[0]
    if first_col == 'game_id':
        train = train.merge(features, on='game_id', how='inner')
        return train
    else:
        shifted = shift_week_number(features, n=1)
        train = map_features_to_games(train, shifted)
        return train


def reduce_training_cols(games, games_cols):
    """Reduce the games data to only the columns that will be used for
    training.
    
    :param pd.DataFrame games: Games data.
    :param list games_cols: Columns to keep.
    :return: Games data with reduced columns.
    :rtype: pd.DataFrame
    """
    games = games[games_cols]
    return games


def build_train(games_cols, raw_games_path, features_path):
    """Build the training data.
    
    :param list games_cols: Columns to keep.
    :param str raw_games_path: Path to raw games data.
    :param str features_path: Path to features directory.
    :return: None
    :rtype: None
    """
    games = pd.read_csv(raw_games_path)
    processed = preprocess_raw_games(games)
    train = reduce_training_cols(processed, games_cols)
    for file_path in walk_features_dir(features_path):
        feature = pd.read_csv(file_path)
        train = merge_feature(train, feature)
    train = (train
             .drop(columns=['away_team', 'home_team', 'week'])
             .sort_values('game_id'))
    return train
