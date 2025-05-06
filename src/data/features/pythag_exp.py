""""""

import polars as pl

from src.data.features.game_stats import extract_game_points
from src.utils import shift_week_number


def calculate_points_for_against(points, side, period):
    """"""
    alias = 'points_for' if side == 'posteam' else 'points_against'
    return (
        points
        .sort(side, 'season', 'week')
        .with_row_index('index')
        .rolling(
            index_column='index',
            group_by=[side, 'season'],
            period=period,
        )
        .agg(
            pl.col('week').last(),
            pl.col('points_game').sum().alias(alias),
        )
    )


def calculate_pythag_stats(points, power=2.77, period='99i', suffix=""):
    """"""
    points_for = calculate_points_for_against(points, 'posteam', period)
    points_against = calculate_points_for_against(points, 'defteam', period)
    return (
        points_for
        .join(
            points_against,
            left_on=['posteam', 'season', 'week'],
            right_on=['defteam', 'season', 'week'],
            how='inner',
        )
        .select(
            pl.col('posteam').alias('team'),
            pl.col('season', 'week'),
            (1 / (1 + (pl.col('points_against') / pl.col('points_for')).pow(power))).alias(f'pythag{suffix}'),
        )
    )


def build_pythag_features(plays):
    """"""
    points = extract_game_points(plays)
    pythag = calculate_pythag_stats(points)
    pythag_recent = calculate_pythag_stats(points, period='5i',
                                           suffix='_recent')
    return (
        pythag
        .join(
            pythag_recent,
            on=['team', 'season', 'week'],
        )
        .pipe(shift_week_number)
    )


if __name__ == '__main__':
    from src.config.config import PATHS
    from src.data.build import clean_plays

    raw_plays_path = PATHS['raw_plays']

    seasons = range(2001, 2025)
    paths = [raw_plays_path / f'play_by_play_{season}.parquet' for season in seasons]
    raw_plays = pl.concat([pl.scan_parquet(p) for p in paths],
                          how='vertical_relaxed')
    plays = clean_plays(raw_plays)

    pythag = build_pythag_features(plays)
    print(pythag.collect())

