"""Utility functions. These functions are used in mutliple places throughout the source code and cannot be coupled to any particular module."""

import os
import requests
import argparse
import datetime

import numpy as np
import pandas as pd


def fix_game_times(games):
    """Cleans the gametime column.

    Impute missing game times based on weekday column. Games on Sunday are
    imputed to start at 13:00. Games on Monday are imputed to start at 20:15.

    Replace 9:00 with 21:00.

    :param games: Raw games dataframe.
    :type games: pd.DataFrame
    :return: Cleaned game times.
    :rtype: pd.Series
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


def walk_features_dir(features_path):
    """Yield every py file path in the features dir.
    
    :param str features_path: path to features directory
    :return: feature file path
    :rtype: str
    """
    for root, dirs, files in os.walk(features_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                yield file_path


def map_team_data_to_games(games, stats, feature_name):
    """Map a pandas series of team stats with a season/week/team multiindex
    to game_ids in the games data.
    
    :param pd.DataFrame games: raw games dataframe
    :param pd.Series stats: series of team stats with season/week/team index
    :return: dataframe with game_id, away_team_stat, home_team_stat
    :rtype: pd.DataFrame
    """
    merged = (games
              .merge(stats, how='inner',
                     left_on=['season', 'week', 'away_team'],
                     right_on=['season', 'week', 'team'])
              .rename(columns={f'{feature_name}': f'away_{feature_name}'})
              .merge(stats, how='inner',
                     left_on=['season', 'week', 'home_team'],
                     right_on=['season', 'week', 'team'])
              .rename(columns={f'{feature_name}': f'home_{feature_name}'}))
    remapped_stats = (merged
                      [['game_id', f'away_{feature_name}', f'home_{feature_name}']]
                      .set_index('game_id'))
    return remapped_stats


def refresh_raw_data(url, save_path):
    """Fetches data from a URL and saves to disk.

    :param str url: URL to fetch data from
    :param str path: Path to save data to
    :return: None
    :rtype: None
    """
    r = requests.get(url)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(r.content)
    return


def get_date_n_days_out(n):
    """Get the date n days from today.
    
    :param int n: number of days from today
    :return: date n days from today in YYYY-MM-DD format
    :rtype: str
    """
    today = datetime.date.today()
    n_days_out = today + datetime.timedelta(days=n)
    formatted_date = n_days_out.strftime('%Y-%m-%d')
    return formatted_date


def shift_week_number(df, n, dropna=True):
    """Shifts the week index of a season/team/week multiindex.
    
    Aggregations are nearly always calculated up to the *end* of a week.
    Shifting is useful when you want your stats to align with the week before
    or the week after (consistent with the bye weeks).

    :param pd.DataFrame df: dataframe with season/team/week multiindex
    :param int n: number of rows to shift
    :param bool dropna: whether to drop rows with missing values
    :return: dataframe with shifted season/team index
    :rtype: pd.DataFrame
    """
    df = (df
          .groupby(['season', 'team'])
          .shift(n))
    if dropna:
        df = df.dropna()
    return df


