"""Helper functions for building engineered features for team stats."""

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


def make_pythag_exp_feature(games, pythag_exp_name, output_dir):
    """Build pythagorean expectation feature.
    
    :param pd.DataFrame games: raw games dataframe
    :param str pythag_exp_name: name of pythagorean expectation feature
    :param str output_dir: path to save stats features
    :return: None
    :rtype: None
    """
    cumulative_points = calculate_cumulative_points(games)
    pythag_exp = calculate_pythag_exp(cumulative_points)
    pythag_exp.name = pythag_exp_name
    pythag_exp = map_team_data_to_games(games, pythag_exp)
    pythag_exp.to_csv(f"{output_dir}/{pythag_exp_name}.csv")
    return 


def make_elo_feature(games, elo_data, elo_name, output_dir):
    """"""
    # fix the chargers (lol as if. go chiefs.)
    elo_data = elo_data.replace('LAC', 'SD')
    elo_data['team1'] = np.where((elo_data['team1'] == 'SD') & (elo_data['season'] > 2016),
                                 'LAC', elo_data['team1'])
    elo_data['team2'] = np.where((elo_data['team2'] == 'SD') & (elo_data['season'] > 2016),
                                 'LAC', elo_data['team2'])
    # fix the rams
    elo_data = elo_data.replace('LAR', 'STL')
    elo_data['team1'] = np.where((elo_data['team1'] == 'STL') & (elo_data['season'] > 2015),
                                 'LA', elo_data['team1'])
    elo_data['team2'] = np.where((elo_data['team2'] == 'STL') & (elo_data['season'] > 2015),
                                 'LA', elo_data['team2'])

    # fix the raiders (lol as if. go chiefs.)
    elo_data['team1'] = np.where((elo_data['team1'] == 'OAK') & (elo_data['season'] > 2019),
                                 'LV', elo_data['team1'])
    elo_data['team2'] = np.where((elo_data['team2'] == 'OAK') & (elo_data['season'] > 2019),
                                 'LV', elo_data['team2'])

    # get elo scores for each game
    games = games.merge(elo_data, how='left',
                        left_on=['gameday', 'away_team', 'home_team'],
                        right_on=['date', 'team2', 'team1'])
    elo_scores = games[['game_id', 'elo2_pre', 'elo1_pre']]
    elo_scores.columns = ['game_id', 'away_elo', 'home_elo']
    elo_scores = (elo_scores
                  .set_index('game_id')
                  .sort_index()
                  .dropna())
    elo_scores.to_csv(f"{output_dir}/{elo_name}.csv")


def build_features(metadata, raw_games_path, raw_elo_path, output_dir, **kwargs):
    """Build engineered features for team stats.
    
    :param dict metadata: metadata for stats features
    :param str raw_games_path: path to raw games data
    :param str raw_elo_path: path to raw elo data
    :param str output_dir: path to save stats features
    :param dict kwargs: additional arguments
    :return: None
    :rtype: None
    """
    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    elo_data = pd.read_csv(raw_elo_path)
    feature_names = list(metadata)
    make_pythag_exp_feature(games, feature_names[0], output_dir)
    make_elo_feature(games, elo_data, feature_names[1], output_dir)
    return
