"""Helper functions for building features to make predictions on."""

import os
import joblib

import numpy as np
import pandas as pd

from src.utils import map_team_data_to_games, walk_features_dir
from src.config.config import (PATHS,
                               TRAINING,
                               CURRENT_SEASON,
                               CURRENT_WEEK)


def get_most_recent_dir(path):
    """Get the largest dir name.
    
    :param path: *str*
        Path to directory.
    :return: *str*
        Path to most recent results dir.
    """
    most_recent_dir = sorted(os.listdir(path))[-1]
    results_path = os.path.join(path, most_recent_dir)
    return results_path


def merge_most_recent_feature(games, feature, current_season, current_week):
    """Merge the most recent feature to the games df.
    
    :param games: *pd.DataFrame of shape (n_games, n_cols)*
        Raw games df.
    :param feature: *pd.DataFrame of shape (n_weeks, n_cols)*
        Feature df. Index should be either game_id or (season, team, week).
    :param current_season: *int*
        Current season.
    :param current_week: *int*
        Current week.
    :return: *pd.DataFrame of shape (n_games, n_cols)*
        Games df with feature merged.
    """
    first_col = feature.columns[0]
    if first_col == 'game_id':
        games = games.merge(feature, on='game_id', how='inner')
        return games
    else:
        feature_name = feature.columns[-1]
        feature = (feature
                   .query('season == @current_season')
                   .groupby('team')
                   .tail(1)
                   .assign(week=current_week)
                   .set_index(['season', 'team', 'week']))
        feature = map_team_data_to_games(games, feature, feature_name)
        games = games.merge(feature, on='game_id', how='inner')
        return games


def build_prediction_data(upcoming, features_path):
    """Build the data to make predictions on.
    
    :param upcoming: *pd.DataFrame of shape (n_games, n_cols)*
        Upcoming games from the raw games df.
    :param features_path: *str*
        Path to features data.
    :return: *pd.DataFrame of shape (n_games, n_cols)*
        Data to make predictions on.
    """
    for file_path in walk_features_dir(features_path):
        feature = pd.read_csv(file_path)
        upcoming = merge_most_recent_feature(upcoming, feature,
                                             CURRENT_SEASON, CURRENT_WEEK)
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


    games = pd.read_csv(raw_games_path, usecols=games_cols)
    q = 'season == @CURRENT_SEASON and week == @CURRENT_WEEK'
    upcoming = games.query(q)
    prediction_data = build_prediction_data(upcoming, features_path)


    most_recent_results = get_most_recent_dir(results_path)
    model = joblib.load(os.path.join(most_recent_results, 'baseline_model.pkl'))
    predictions = model.predict_proba(prediction_data)
    prediction_data[['baseline_away_win_prob', 'baseline_home_win_prob']] = predictions

    model = joblib.load(os.path.join(most_recent_results, 'swift_model.pkl'))
    predictions = model.predict_proba(prediction_data)
    prediction_data[['swift_away_win_prob', 'swift_home_win_prob']] = predictions


    prediction_data.columns = (prediction_data.columns
                               .str.replace('obj', 'home')
                               .str.replace('adv', 'away'))
    prediction_data = prediction_data.drop(columns='home_team_is_home')
    prediction_data = prediction_data.round(3)


    os.makedirs(prediction_data_path, exist_ok=True)
    filename = f'{CURRENT_SEASON}_week_{CURRENT_WEEK}_predictions.csv'
    prediction_data.to_csv(f'{prediction_data_path}/{filename}', index=False)
