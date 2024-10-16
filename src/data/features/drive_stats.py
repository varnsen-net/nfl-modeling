""""""

import numpy as np
import pandas as pd

from src.config.config import CURRENT_SEASON
from src.data.features.helpers import build_adjusted_data_for_season


def pipe_print(df):
    print(df)
    return df


def preprocess_drive_data(drives):
    """"""
    start_sides = (drives['drive_start_yard_line']
                   .str[:3]
                   .rename('drive_start_side'))
    # we do a concat here because the usual column assignment method caused an
    # annoying warning about fragmented dataframes
    drives = pd.concat([drives, start_sides], axis=1)
    drives = drives.query('drive_start_side == posteam')
    return drives


def extract_drive_results(raw_plays):
    """Transform play-by-play data into drive-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Drive-level data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'week', 'defteam', 'fixed_drive'])
            .first()
            ['fixed_drive_result']
            .map({'Touchdown': 7, 'Field goal': 3})
            .fillna(0)
            .groupby(['posteam', 'week', 'defteam'])
            .agg(['sum', 'count']))


def build_drive_features(raw_plays_path):
    """Build features based on drive-level data.

    :param pathlib.Path raw_plays_path: Path to raw play-by-play data.
    :return: Drive-level features.
    :rtype: pd.DataFrame
    """
    features = pd.DataFrame()
    stat_name = 'points_drive'
    for season in list(range(1999, CURRENT_SEASON + 1)):
        print(f"Processing season {season}")
        raw_plays = raw_plays_path / f"play_by_play_{season}.parquet"
        raw_plays = pd.read_parquet(raw_plays)
        raw_plays = raw_plays.query('posteam != "" and posteam != "None"') # move this to a preprocessing step
        drives = extract_drive_results(raw_plays)
        drives.columns = ['value', 'count']
        max_week = drives.index.get_level_values('week').max()
        week_nums = range(1, max_week + 1)
        season_features = build_adjusted_data_for_season(drives,
                                                         stat_name,
                                                         week_nums)
        season_features['season'] = season
        features = pd.concat([features, season_features])
    return features


if __name__ == '__main__':
    from src.config.config import PATHS
    from src.data.features.helpers import (create_weekly_csums,
                                           calculate_league_avgs,
                                           calculate_adversary_mscores,
                                           calculate_adjusted_scores,
                                           format_adj_mscores)

    raw_plays = pd.read_parquet(PATHS['raw_plays'] / 'play_by_play_2024.parquet')
    raw_plays = raw_plays.query('posteam != "" and posteam != "None"')

    drives = extract_drive_results(raw_plays)
    drives.columns = ['value', 'count']

    stat_name = 'points_drive'
    obj_side = 'posteam'
    adv_side = 'defteam' if obj_side == 'posteam' else 'posteam'

    drives = (drives
              .droplevel(adv_side)
              .unstack(obj_side))
    print(drives['value'])
    print(drives['count'])
    values_sum = drives['value'].sum().sum()
    counts_sum = drives['count'].sum().sum()
    league_mean = values_sum / counts_sum
    game_means = drives['value'] / drives['count']
    league_mad = (game_means - league_mean).abs().median().median()
    print(league_mean, league_mad)
    game_mscores = (game_means - league_mean) / league_mad
    print(game_mscores)
    print(game_mscores['KC'])


    # adv_csums = create_weekly_csums(drives, obj_side, adv_side)
    # adv_means = adv_csums['value'] / adv_csums['count']
    # league_mean, league_mad = calculate_league_avgs(adv_csums, adv_means)

    # adv_mscores = calculate_adversary_mscores(adv_means, league_mean,
                                              # league_mad, adv_side)
    # adj_mscores = calculate_adjusted_scores(drives, adv_mscores, league_mean,
                                            # league_mad, adv_side)
    # adj_mscores = format_adj_mscores(adj_mscores, obj_side, stat_name, 5)
    # print(adj_mscores)
