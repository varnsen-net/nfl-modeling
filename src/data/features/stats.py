import json

import pandas as pd
import numpy as np

from src.utils import parse_common_args, map_team_data_to_games


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
    :param dict stats_features: stats features to build
    :param str output_path: path to save stats features
    :return: None
    :rtype: None
    """
    cumulative_points = calculate_cumulative_points(games)
    pythag_exp = calculate_pythag_exp(cumulative_points)
    pythag_exp.name = pythag_exp_name
    pythag_exp = map_team_data_to_games(games, pythag_exp)
    return pythag_exp


def make_stats_features(games, stats_features, output_path):
    """Build engineered features for team stats.
    
    :param pd.DataFrame games: raw games dataframe
    :param dict stats_features: stats features to build
    :param str output_path: path to save stats features
    :return: None
    :rtype: None
    """
    feature_names = list(stats_features)
    pythag_exp_name = feature_names[0]
    pythag_exp = make_pythag_exp_feature(games, pythag_exp_name)
    pythag_exp.to_csv(f"{output_path}/{pythag_exp_name}.csv")
    return


if __name__ == '__main__':
    args = parse_common_args()
    config_path = args.c
    raw_games_path = args.g
    output_path = args.o

    with open(config_path, 'r') as f:
        config = json.load(f)
        stats_features = config['features']['stats']

    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    make_stats_features(games, stats_features, output_path)
             

