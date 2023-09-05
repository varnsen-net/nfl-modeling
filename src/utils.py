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


def parse_common_args():
    """Parse a common set of arguments for writing data files."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', help='path to config file')
    argparser.add_argument('-g', help='path to raw games data')
    argparser.add_argument('-w', help='path to raw weather data')
    argparser.add_argument('-cc', help='path to city coordinates')
    argparser.add_argument('-o', help='path to output files')
    args = argparser.parse_args()
    return args

