""""""

import numpy as np
import pandas as pd

from src.data.features.helpers import build_adjusted_features
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
            .groupby(['season', 'posteam', 'week', 'defteam', 'fixed_drive'])
            .first()
            [['fixed_drive_result', 'absolute_yrdln', 'posteam_type']]
            .reset_index()
            .merge(exp_values, how='left', on=['absolute_yrdln', 'posteam_type'])
            .set_index(['season', 'posteam', 'week', 'defteam', 'fixed_drive'])
            .assign(fixed_drive_result=remap_drive_results,
                    adj_fixed_drive_result=calculate_value_over_exp)
            ['adj_fixed_drive_result']
            .groupby(['posteam', 'season', 'week', 'defteam'])
            .agg(['sum', 'count']))


def format_col_names(col_names):
    """"""
    return ['points_drive', 'count']


def build_drive_features(raw_plays, exp_values):
    """Build features based on drive-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Drive-level features.
    :rtype: pd.DataFrame
    """
    # think about moving this processing step elsewhere
    raw_plays = (raw_plays
                 .query('yrdln.notnull()')
                 .assign(yrdln_fixed=fix_bad_yardlines,
                         absolute_yrdln=make_absolute_yardlines))
    raw_features = extract_drive_results(raw_plays, exp_values)
    raw_features.columns = format_col_names(raw_features.columns)
    adjusted_features = build_adjusted_features(raw_features)
    return adjusted_features


if __name__ == '__main__':
    from src.data.ancillary.expectations import preprocess_plays
    from src.config.config import PATHS

    raw_plays_path = PATHS['raw_plays']
    expected_values_path = PATHS['expected_values']

    seasons = list(range(2021, 2025))
    plays = pd.concat([pd.read_parquet(raw_plays_path / f"play_by_play_{season}.parquet")
                           for season in seasons])
    exp_values = pd.read_csv(expected_values_path)

    drive_features = build_drive_features(plays, exp_values)
    print(drive_features)
