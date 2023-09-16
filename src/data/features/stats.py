import json

import pandas as pd
import numpy as np

from src.utils import map_team_data_to_games


def calculate_cumulative_points(games):
    """Calculate the cumulative points for/against each team up to each week.
    
    :param pd.DataFrame games: raw games dataframe
    :return: cumulative points for/against each team *up to* the indexed week
    :rtype: pd.DataFrame
    """
    home_teams = games[['season', 'week', 'home_team', 'home_score', 'away_score']]
    away_teams = games[['season', 'week', 'away_team', 'away_score', 'home_score']]
    points = pd.DataFrame(np.append(home_teams, away_teams, axis=0))
    points.columns = ['season', 'week', 'team', 'points_for', 'points_against']
    points = (points
              .astype({'points_for': int, 'points_against': int})
              .set_index(['season', 'team', 'week'])
              .sort_index())
    points['cpf'] = points.groupby(['season', 'team']).cumsum()['points_for']
    points['cpa'] = points.groupby(['season', 'team']).cumsum()['points_against']
    points = (points
              .groupby(['season', 'team'])
              .shift(1)
              .dropna())
    return points


def calculate_pythag_exp(points):
    """Calculates the pythagorean expectation for a set of points for and
    points against.
    
    :param pd.DataFrame points: df with points for and points against columns
    :return: pythagorean expectations
    :rtype: pd.Series
    """
    numerator = points['cpf'] ** 2.68
    denominator = points['cpf'] ** 2.68 + points['cpa'] ** 2.68
    pythag_exp = numerator / denominator
    return pythag_exp


def make_pythag_exp_feature(games, pythag_exp_name):
    """Build pythagorean expectation feature.
    
    :param pd.DataFrame games: raw games dataframe
    :param str pythag_exp_name: name of pythagorean expectation feature
    :return: None
    :rtype: None
    """
    cumulative_points = calculate_cumulative_points(games)
    pythag_exp = calculate_pythag_exp(cumulative_points)
    pythag_exp.name = pythag_exp_name
    pythag_exp = map_team_data_to_games(games, pythag_exp)
    return pythag_exp


def build_features(metadata, raw_games_path, output_dir, **kwargs):
    """Build engineered features for team stats.
    
    :param dict metadata: metadata for stats features
    :param str raw_games_path: path to raw games data
    :param str output_dir: path to save stats features
    :param dict kwargs: additional arguments
    :return: None
    :rtype: None
    """
    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    feature_names = list(metadata)
    pythag_exp_name = feature_names[0]
    pythag_exp = make_pythag_exp_feature(games, pythag_exp_name)
    pythag_exp.to_csv(f"{output_dir}/{pythag_exp_name}.csv")
    return
