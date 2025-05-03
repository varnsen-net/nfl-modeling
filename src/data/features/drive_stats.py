"""
"""
# TODO: pythagorean expectation
# TODO: home/away splits
# TODO: normal game situations

import polars as pl

from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number


def get_drive_result():
    """"""
    return (
        pl.first('fixed_drive_result')
        .replace_strict({'Touchdown': 7, 'Field goal': 3}, default=0)
        .alias('points_drive')
    )


def extract_drive_points(plays):
    """Extract the results for all drives.

    Example output:

    | posteam | season | week | defteam | points_game | count |
    | ---     | ---    | ---  | ---     | ---         | ---   |
    | str     | i32    | i32  | str     | i64         | u32   |
    |---------|--------|------|---------|-------------|-------|
    | CAR     | 2022   | 7    | TB      | 21          | 10    |
    | ATL     | 2021   | 11   | NE      | 0           | 11    |
    | BUF     | 2023   | 8    | TB      | 24          | 11    |
    | ...     | ...    | ...  | ...     | ...         | ...   |
    | BUF     | 2023   | 3    | WAS     | 30          | 11    |
    | WAS     | 2023   | 11   | NYG     | 20          | 14    |
    | IND     | 2023   | 7    | CLE     | 38          | 16    |

    :param pl.DataFrame plays: The plays DataFrame.
    :return: The results for all drives.
    :rtype: pl.DataFrame
    """
    return (
        plays.group_by(
            ['season', 'week', 'posteam', 'fixed_drive']
        )
        .agg(
            get_drive_result(),
            pl.col('defteam').first()
        )
        .group_by(
            ['season', 'week', 'posteam']
        )
        .agg(
            pl.col('defteam').first(),
            pl.col('points_drive').sum(),
            pl.col('points_drive').len().alias('count')
        )
    )


def build_drive_stats_features(plays):
    """"""
    return (
        plays
        .pipe(extract_drive_points)
        .pipe(build_adjusted_features)
        .pipe(shift_week_number)
    )
