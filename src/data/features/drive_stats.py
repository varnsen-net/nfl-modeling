""""""

import numpy as np
import pandas as pd

from src.data.features.helpers import build_adjusted_data_for_season
from src.data.ancillary.expectations import (fix_bad_yardlines,
                                             make_absolute_yardlines)


def remap_drive_results(drives):
    """"""
    return (drives
            ['fixed_drive_result']
            .map({'Touchdown': 7, 'Field goal': 3})
            .fillna(0))


def calculate_value_over_exp(drives):
    """"""
    return drives['fixed_drive_result'] - drives['expected_value']


def extract_drive_results(raw_plays, exp_values):
    """Transform play-by-play data into drive-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Drive-level data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'week', 'defteam', 'fixed_drive'])
            .first()
            [['fixed_drive_result', 'absolute_yrdln', 'posteam_type']]
            .reset_index()
            .merge(exp_values, how='left', on=['absolute_yrdln', 'posteam_type'])
            .set_index(['posteam', 'week', 'defteam', 'fixed_drive'])
            .assign(fixed_drive_result=remap_drive_results,
                    adj_fixed_drive_result=calculate_value_over_exp)
            ['adj_fixed_drive_result']
            .groupby(['posteam', 'week', 'defteam'])
            .agg(['sum', 'count']))


def build_drive_features(raw_plays, exp_values, season):
    """Build features based on drive-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :param int season: Season year.
    :return: Drive-level features.
    :rtype: pd.DataFrame
    """
    # think about moving this processing step elsewhere
    raw_plays = (raw_plays
                 .query('yrdln.notnull()')
                 .assign(yrdln_fixed=fix_bad_yardlines,
                         absolute_yrdln=make_absolute_yardlines))

    points = extract_drive_results(raw_plays, exp_values)
    max_week = points.index.get_level_values('week').max()
    week_nums = range(1, max_week + 1)

    stat_name = 'points_drive'
    features = build_adjusted_data_for_season(points, stat_name,
                                              week_nums, 'mscores')
    features['season'] = season
    return features


if __name__ == '__main__':
    from src.data.ancillary.expectations import preprocess_plays
    from src.config.config import PATHS

    raw_plays_path = PATHS['raw_plays']
    expected_values_path = PATHS['expected_values']

    season = 2023
    plays = pd.read_parquet(raw_plays_path / f"play_by_play_{season}.parquet")
    plays = preprocess_plays(plays)
    exp_values = pd.read_csv(expected_values_path)

    drive_features = build_drive_features(plays, exp_values, season)
    print(drive_features)
