""""""

import os

import polars as pl


pl.Config.set_tbl_formatting("ASCII_MARKDOWN")


def clean_plays(plays):
    """Do some basic cleaning of the plays dataframe.

    :param pl.LazyFrame plays: The plays dataframe.
    :return: A cleaned plays dataframe.
    :rtype: pl.LazyFrame
    """
    return plays.filter(
        pl.col('posteam').is_not_null(),
        pl.col('posteam') != "",
        pl.col('season_type') == "REG",
        pl.col('location') == "Home",
    )


def reduce_to_normal_plays(plays):
    """Reduce the plays dataframe to only include normal plays.

    Normal plays are defined as those where the score and clock do not have
    a significant impact on playcalling.

    :param pl.LazyFrame plays: The plays dataframe.
    :return: A reduced plays dataframe.
    :rtype: pl.LazyFrame
    """
    return plays.filter(
        pl.col('half_seconds_remaining') > 120,
        pl.col('score_differential').abs() < 16,
    )


def reduce_raw_games(games, games_cols, min_year):
    """Reduce the games dataframe using polars.
    
    :param pl.LazyFrame games: The games dataframe.
    :param list games_cols: The columns to keep in the games dataframe.
    :param int min_year: The minimum year to keep in the games dataframe.
    :return: A reduced games dataframe.
    :rtype: pl.LazyFrame
    """
    return (
        games
        .filter(
            pl.col('week') > 4,
            pl.col('season') >= min_year,
            ~((pl.col('week') > 16) & (pl.col('season') < 2021)),
            ~((pl.col('week') > 17) & (pl.col('season') >= 2021))
        )
        .drop(['game_type', 'location'])
        .drop_nulls('result')
        .select(games_cols)
    )


def rename_cols(col_name):
    """Rename columns to specify home vs away team feature.

    :param str col_name: The column name to rename.
    :return: The renamed column name.
    :rtype: str
    """
    if col_name.endswith(('_posteam', '_defteam')):
        return col_name + '_away'
    elif col_name.endswith('_right'):
        return col_name.replace('_right', '_home')
    else:
        return col_name


if __name__ == '__main__':
    from src.data.raw.games import refresh_games_data
    from src.data.raw.plays import refresh_plays_data
    from src.data.features.travel import build_travel_features
    from src.data.features.drive_stats import build_drive_stats_features
    from src.data.features.game_stats import build_game_stats_features
    from src.config.config import (TRAINING,
                                   CURRENT_SEASON,
                                   CURRENT_WEEK,
                                   RAW_DATA_URLS,
                                   PATHS)

    raw_games_path = PATHS['raw_games']
    raw_plays_path = PATHS['raw_plays']
    city_coords_path = PATHS['city_coordinates']
    expected_values = PATHS['expected_values']
    features_path = PATHS['features']
    train_path = PATHS['train']
    min_year = TRAINING['min_year']
    holdout_year_start = TRAINING['holdout_year_start']
    games_cols = TRAINING['games_cols']
    games_url = RAW_DATA_URLS['games']
    plays_url = RAW_DATA_URLS['plays']

    os.makedirs(features_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    print('Refreshing raw games data...')
    refresh_games_data(games_url, raw_games_path)

    print('Refreshing raw play-by-play data...')
    refresh_plays_data(CURRENT_SEASON, plays_url, raw_plays_path)

    print('Loading and processing raw data...')
    seasons = range(2001, CURRENT_SEASON + 1)
    paths = [raw_plays_path / f'play_by_play_{season}.parquet' for season in seasons]
    raw_plays = pl.concat([pl.scan_parquet(p) for p in paths],
                          how='vertical_relaxed')
    plays = clean_plays(raw_plays)
    raw_games = pl.scan_csv(raw_games_path)
    games = reduce_raw_games(raw_games, games_cols, min_year)
    city_coords = pl.scan_csv(city_coords_path)

    print('Building features...')
    features = (
        games
        .join(
            build_travel_features(raw_games, city_coords),
            on='game_id',
            how='inner',
        )
        .join(
            build_game_stats_features(plays),
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='inner',
        )
        .join(
            build_game_stats_features(plays),
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='inner',
        )
        .join(
            build_drive_stats_features(plays),
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='inner',
        )
        .join(
            build_drive_stats_features(plays),
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='inner',
        )
        .rename(rename_cols)
        .sort('game_id')
    )
    features.sink_csv(features_path / 'features.csv')

    print('Building training and test sets...')
    all_training_data = (
        features
        .with_columns(
            pl.when(pl.col('result') > 0).then(1).otherwise(0).alias('target'),
            pl.when(pl.col('season') >= holdout_year_start).then(1).otherwise(0).alias('holdout'),
        )
        .drop(['week', 'away_team', 'home_team', 'result'])
    )
    train = all_training_data.filter(pl.col('holdout') == 0).drop('holdout')
    test = all_training_data.filter(pl.col('holdout') == 1).drop('holdout')
    train.sink_csv(train_path / 'train.csv')
    test.sink_csv(train_path / 'test.csv')
