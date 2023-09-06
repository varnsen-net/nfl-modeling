import os
import json
import argparse

import numpy as np
import pandas as pd


def preprocess_games_data(games):
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', help='path to config file')
    argparser.add_argument('-g', help='path to games data')
    argparser.add_argument('-f', help='path to features data')
    argparser.add_argument('-o', help='path to output data')
    args = argparser.parse_args()
    config_path = args.c
    raw_games_path = args.g
    features_path = args.f
    output_path = args.o

    with open(config_path, 'r') as f:
        config = json.load(f)
        games_cols = config['training']['games_cols']

    games = pd.read_csv(raw_games_path)
    processed = preprocess_games_data(games)
    reduced = reduce_training_cols(processed, games_cols)
    train = merge_features(reduced, features_path)
    train.to_csv(output_path, index=False)
