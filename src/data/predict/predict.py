"""Helper functions for building features to make predictions on."""

import os
import joblib

import numpy as np
import pandas as pd

from src.utils import map_features_to_games, walk_features_dir
from src.config.config import (PATHS,
                               TRAINING,
                               CURRENT_SEASON,
                               CURRENT_WEEK)


def get_most_recent_dir(path):
    """Get the largest dir name. Assumes that the dir names are integer
    timestamps.
    
    :param str path: Path to the parent dir to search.
    :return: Path to the most recent dir.
    :rtype: str
    """
    most_recent_dir = sorted(os.listdir(path))[-1]
    results_path = os.path.join(path, most_recent_dir)
    return results_path


def merge_most_recent_feature(games, feature, current_season, current_week):
    """Merge the most recent feature to the games df.
    
    :param games: Raw games df.
    :type games: pd.DataFrame of shape (n_games, n_cols)
    :param feature: Feature df. Index should be either game_id or season/team/week.
    :type feature: pd.DataFrame of shape (n_weeks, n_cols)
    :param int current_season: Current NFL season.
    :param int current_week: Current NFL week.
    :return: Games df with the most recent feature merged.
    :rtype: pd.DataFrame of shape (n_games, n_cols)
    """
    first_col = feature.columns[0]
    if first_col == 'game_id':
        games = games.merge(feature, on='game_id', how='inner')
        return games
    else:
        feature = (feature
                   .query('season == @current_season')
                   .groupby('team')
                   .tail(1)
                   .assign(week=current_week))
        games = map_features_to_games(games, feature)
        return games


def build_prediction_data(upcoming, features_path):
    """Build the data to make predictions on.
    
    :param upcoming: Upcoming games df.
    :type upcoming: pd.DataFrame of shape (n_games, n_cols)
    :param str features_path: Path to the features dir.
    :return: Data to make predictions on.
    :rtype: pd.DataFrame of shape (n_games, n_cols)
    """
    for file_path in walk_features_dir(features_path):
        feature = pd.read_csv(file_path)
        upcoming = merge_most_recent_feature(upcoming, feature,
                                             CURRENT_SEASON, CURRENT_WEEK)
    upcoming = (upcoming
                .drop(columns=['away_team', 'home_team', 'season', 'week'])
                .set_index('game_id')
                .sort_index())
    upcoming.columns = (upcoming.columns
                        .str.replace('home', 'obj')
                        .str.replace('away', 'adv'))
    upcoming['obj_team_is_home'] = 1
    return upcoming


if __name__ == "__main__":
    raw_games_path = PATHS['raw_games']
    features_path = PATHS['features']
    games_cols = TRAINING['games_cols']
    results_path = PATHS['results']
    prediction_data_path = PATHS['prediction']


    games = pd.read_csv(raw_games_path)
    upcoming = (games
                .loc[games['result'].isnull()]
                .filter(games_cols))
    prediction_data = build_prediction_data(upcoming, features_path)
    readable_data = prediction_data.copy()

    most_recent_results = get_most_recent_dir(results_path)
    model = joblib.load(os.path.join(most_recent_results, 'baseline_model.pkl'))
    predictions = model.predict_proba(prediction_data)
    readable_data[['baseline_away_win_prob', 'baseline_home_win_prob']] = predictions

    model = joblib.load(os.path.join(most_recent_results, 'svc_model.pkl'))
    predictions = model.predict_proba(prediction_data)
    readable_data[['swift_away_win_prob', 'swift_home_win_prob']] = predictions

    readable_data.columns = (readable_data.columns
                             .str.replace('obj', 'home')
                             .str.replace('adv', 'away'))
    readable_data = readable_data.round(3)

    os.makedirs(prediction_data_path, exist_ok=True)
    filename = f'{CURRENT_SEASON}_week_{CURRENT_WEEK}_predictions.csv'
    readable_data.to_csv(f'{prediction_data_path}/{filename}')
