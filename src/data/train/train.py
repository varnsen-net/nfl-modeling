"""Helper functions for building the training data."""

import os

import numpy as np
import pandas as pd

from src.utils import (shift_week_number,
                       map_team_data_to_games,
                       walk_features_dir)


def preprocess_raw_games(games):
    """Reduce the games dataframe -- only consider regular season games played
    at home.
    
    :param pd.DataFrame games: raw games dataframe
    :return: reduced games data
    :rtype: pd.DataFrame
    """
    games = (games
             .loc[games['week'] > 4]
             .loc[games['game_type'] == 'REG']
             .loc[games['location'] == 'Home']
             .drop(columns=['game_type', 'location'])
             .dropna(subset=['result'])
             .copy())
    return games


def merge_feature(train, feature):
    """Merge a feature into the training data.
    
    :param pd.DataFrame train: training data
    :param pd.DataFrame feature: feature data
    :return: training data with feature merged in
    :rtype: pd.DataFrame
    """
    first_col = feature.columns[0]
    if first_col == 'game_id':
        train = train.merge(feature, on='game_id', how='inner')
        return train
    else:
        feature_name = feature.columns[-1]
        feature = feature.set_index(['season', 'team', 'week'])
        shifted = shift_week_number(feature, n=1)
        feature = map_team_data_to_games(train, shifted, feature_name)
        train = train.merge(feature, on='game_id', how='inner')
        return train


def reduce_training_cols(games, games_cols):
    """Reduce the games data to only the columns that will be used for
    training.
    
    :param pd.DataFrame games: games data
    :param list games_cols: columns to keep
    :return: games data with reduced columns
    :rtype: pd.DataFrame
    """
    games = games[games_cols]
    return games


def build_train(games_cols, raw_games_path, features_path):
    """Build the training data.
    
    :param list games_cols: columns to keep
    :param str raw_games_path: path to raw games data
    :param str features_path: path to features directory
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
             .drop(columns=['away_team', 'home_team'])
             .sort_values('game_id'))
    return train
