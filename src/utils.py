"""Utility functions. These functions are used in mutliple places throughout the source code and cannot be coupled to any particular module."""

import argparse

import numpy as np
import pandas as pd


def fix_game_times(games):
    """Cleans the gametime column.

    - Impute missing game times based on weekday column. Games on Sunday are
    imputed to start at 13:00. Games on Monday are imputed to start at 20:15.
    - Replace 9:00 with 21:00.

    :param pd.DataFrame games: sharpe games dataframe
    :return: games dataframe with imputed game times
    :rtype: pd.DataFrame
    """
    times = games[['gametime', 'weekday']].copy()
    filter = times['gametime'].isna()
    missing_gametimes = times.loc[filter]
    imputed_gametimes = np.where(missing_gametimes['weekday'] == 'Sunday',
                                 '14:45', '20:15')
    times.loc[filter, 'gametime'] = imputed_gametimes
    game_times = times['gametime'].replace('09:00', '21:00')
    return game_times


def get_kickoff_hours(gametimes):
    """Extract the hour when kickoff takes place.

    :param pd.Series gametimes: a column of game times
    :return: kickoff hours
    :rtype: pd.Series
    """
    hours = gametimes.str.split(':').str[0].astype(int)
    return hours


def map_team_data_to_games(games, stats):
    """Map a pandas series of team stats with a season/week/team multiindex
    to game_ids in the games data.
    
    :param pd.DataFrame games: raw games dataframe
    :param pd.Series stats: series of team stats with season/week/team index
    :return: dataframe with game_id, away_team_stat, home_team_stat
    :rtype: pd.DataFrame
    """
    name = stats.name
    merged = (games
              .merge(stats, how='inner',
                     left_on=['season', 'week', 'away_team'],
                     right_on=['season', 'week', 'team'])
              .rename(columns={f'{name}': f'away_{name}'})
              .merge(stats, how='inner',
                     left_on=['season', 'week', 'home_team'],
                     right_on=['season', 'week', 'team'])
              .rename(columns={f'{name}': f'home_{name}'}))
    remapped_stats = (merged
                      [['game_id', f'away_{name}', f'home_{name}']]
                      .set_index('game_id'))
    return remapped_stats


def collect_setup_args():
    """Collects the command line arguments for the setup script.
    
    :return: command line arguments
    :rtype: argparse.Namespace
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', help='path to config file')
    argparser.add_argument('-g', help='path to raw games data')
    argparser.add_argument('-e', help='path to raw elo data')
    argparser.add_argument('-w', help='path to raw weather data')
    argparser.add_argument('-p', help='path to raw play-by-play data')
    argparser.add_argument('-cc', help='path to city coordinates')
    argparser.add_argument('-f', help='path to features data')
    argparser.add_argument('-tr', help='path to training data')
    argparser.add_argument('-te', help='path to test data')
    argparser.add_argument('-r', help='path to training run results')
    args = argparser.parse_args()
    return args
