"""Helper functions for building the training data."""

import os

import numpy as np
import pandas as pd


def preprocess_raw_games(games):
    """Reduce the games dataframe -- only consider regular season games played
    at home.
    
    :param pd.DataFrame games: raw games dataframe
    :return: reduced games data
    :rtype: pd.DataFrame
    """
    games = (games
             .loc[games['game_type'] == 'REG']
             .loc[games['location'] == 'Home']
             .drop(columns=['game_type', 'location'])
             .dropna(subset=['result'])
             .copy())
    return games


def merge_features(games, features_path):
    """Find every csv file in the features directory (and subdirectories) and
    merge them with the games dataframe.
    
    :param pd.DataFrame games: raw games data
    :param str features_path: path to features directory
    :return: games data with features
    :rtype: pd.DataFrame
    """
    for root, dirs, files in os.walk(features_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                features = pd.read_csv(file_path)
                games = games.merge(features, on='game_id', how='inner')
    return games


def make_target_col(games):
    """Make target column for games data.
    
    :param pd.DataFrame games: games data
    :return: games data with target column
    :rtype: pd.DataFrame
    """
    target = np.where(games['result'] > 0, 1, 0)
    return target


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


def build_train(games_cols, raw_games_path, features_path, train_path):
    """Build the training data.
    
    :param list games_cols: columns to keep
    :param str raw_games_path: path to raw games data
    :param str features_path: path to features directory
    :param str train_path: path to save training data
    :return: None
    :rtype: None
    """
    games = pd.read_csv(raw_games_path)
    processed = preprocess_raw_games(games)
    reduced = reduce_training_cols(processed, games_cols)
    train = merge_features(reduced, features_path)
    train.to_csv(train_path, index=False)
    return
