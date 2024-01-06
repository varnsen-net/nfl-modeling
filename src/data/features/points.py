"""Helper functions for building engineered features for team points."""

import pandas as pd
import numpy as np


def reframe_team_as_opponent(df):
    """Transform a df index such that 'team' is now 'opponent'.
    
    :param df: Dataframe with season/team/week multiindex.
    :type df: pd.DataFrame of shape (n_samples, n_features)
    :return: Dataframe with season/opponent/week multiindex.
    :rtype: pd.DataFrame of shape (n_samples, n_features)
    """
    df = df.droplevel('opponent')
    df.index = df.index.rename({'team': 'opponent'})
    df = df.add_prefix('opp_')
    return df


def make_base_points_data(games):
    """Build base points data (e.g. points for and against) for feature
    creation.
    
    :param games: Raw games data.
    :type games: pd.DataFrame of shape (n_games, n_features)
    :return: Points for/against indexed by season, team, and week.
    :rtype: pd.DataFrame of shape (n_samples, 5)
    """
    home_teams = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']]
    away_teams = games[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']]
    points = pd.DataFrame(np.append(home_teams, away_teams, axis=0))
    points.columns = ['season', 'week', 'team', 'opponent', 'points_for', 'points_against']
    points = (points
              .astype({'points_for': int, 'points_against': int})
              .set_index(['season', 'team', 'week', 'opponent'])
              .sort_index())
    return points


def calculate_avgs(points, window):
    """Calculate averages per team per week per season with expanding or
    rolling windows.
    
    :param points: Points for/against indexed by season, team, and week.
    :type points: pd.DataFrame of shape (n_samples, n_features)
    :param int window: Number of weeks to use for rolling averages.
    :return: Points per game averages indexed by season, team, and week.
    :rtype: pd.DataFrame of shape (n_samples, n_features)
    """
    group = points.groupby(['season', 'team'])
    if window:
        avgs = (group
                .rolling(window=window)
                .mean()
                .add_suffix(f"_avg_{window}wk"))
    else:
        avgs = (group
                .expanding()
                .mean()
                .add_suffix('_avg'))
    avgs = avgs.droplevel([0,1])
    return avgs


def adjust_for_opponent(base, opponent):
    """Adjust base data for opponent strength.
    
    :param base: Base data to adjust. Must have season/team/week/opponent multiindex.
    :type base: pd.DataFrame of shape (n_samples, n_features)
    :param opponent: Opponent data to use for adjustment. Must have season/team/week/opponent multiindex.
    :type opponent: pd.DataFrame of shape (n_samples, n_features)
    :return: Adjusted data.
    :rtype: pd.DataFrame of shape (n_samples, n_features) 
    """
    opponent = (opponent
                .groupby(['season', 'team'])
                .shift(1)
                .fillna(0))
    opponent = reframe_team_as_opponent(opponent)
    merged = pd.merge(base, opponent, left_index=True, right_index=True)
    base_cols = base.columns
    opp_cols = opponent.columns
    adjusted = merged.loc[:,base_cols] - merged.loc[:,opp_cols].values
    adjusted = adjusted.add_prefix('opp_adj_')
    return adjusted


def build_points_features(raw_games_path, window):
    """Build all points features (e.g. net points).
    
    :param str raw_games_path: Path to raw games data.
    :param int window: Number of weeks to use for rolling averages.
    """
    games = pd.read_csv(raw_games_path)
    games = games.dropna(subset=['result'])
    base_points = make_base_points_data(games)
    net_ppg_avgs = calculate_avgs(base_points, window)
    net_ppg_avgs = net_ppg_avgs.iloc[:,[1,0]]
    adj_points = adjust_for_opponent(base_points, net_ppg_avgs)
    adj_points['opp_adj_points_net'] = (adj_points['opp_adj_points_for']
                                        - adj_points['opp_adj_points_against'])
    adj_points = calculate_avgs(adj_points, window)
    adj_points = (adj_points
                  .round(1)
                  .droplevel('opponent')
                  .swaplevel('week', 'team')
                  .reset_index()
                  .sort_values(['season', 'team', 'week'])
                  .set_index(['season', 'team', 'week']))
    return adj_points


