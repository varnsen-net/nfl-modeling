"""
"""
# TODO: pythagorean expectation
# TODO: home/away splits
# TODO: normal game situations

import polars as pl

from src.data.features.scaler import build_adjusted_features
from src.utils import shift_week_number


def extract_game_points(plays):
    """Extract the total points per game for each team.
    
    Example output:

    | posteam | season | week | defteam | points_game | count |
    | ---     | ---    | ---  | ---     | ---         | ---   |
    | str     | i32    | i32  | str     | f64         | i32   |
    |---------|--------|------|---------|-------------|-------|
    | TEN     | 2021   | 1    | ARI     | 13.0        | 1     |
    | ARI     | 2021   | 1    | TEN     | 38.0        | 1     |
    | LV      | 2021   | 1    | BAL     | 33.0        | 1     |
    | ...     | ...    | ...  | ...     | ...         | ...   |
    | NYJ     | 2024   | 14   | MIA     | 26.0        | 1     |
    | ARI     | 2024   | 14   | SEA     | 18.0        | 1     |
    | SEA     | 2024   | 14   | ARI     | 27.0        | 1     |

    :param pl.DataFrame plays: The plays DataFrame.
    :return: The total points per game for each team.
    :rtype: pl.DataFrame
    """
    return (
        plays
        .group_by('posteam', 'season', 'week', 'defteam')
        .agg(pl.last('posteam_score').alias('points_game'))
        .with_columns(count=1)
    )


def build_game_stats_features(plays):
    """"""
    return (
        plays
        .pipe(extract_game_points)
        .pipe(build_adjusted_features)
        .pipe(shift_week_number)
    )
