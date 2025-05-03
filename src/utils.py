"""Utility functions. These functions are used in mutliple places throughout the source code and cannot be coupled to any particular module."""

import requests
import datetime

import numpy as np
import pandas as pd
import polars as pl


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


def shift_week_number(df):
    """Shifts the week number of a df with season/team/week columns.
    
    Aggregations are nearly always calculated up to the *end* of a week.
    Shifting is useful when you want your stats to align with the week before
    or the week after (consistent with the bye weeks).

    :param pd.DataFrame df: Dataframe with season/team/week cols
    :param int n: Number of rows to shift.
    :return: Dataframe with shifted weeks.
    :rtype: pd.DataFrame
    """
    return (
        df
        .sort('team', 'season', 'week')
        .with_columns(
            pl.exclude('team', 'season', 'week').shift(1).over(["team", "season"])
        )
    )
