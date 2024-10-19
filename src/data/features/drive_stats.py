""""""

import numpy as np
import pandas as pd

from src.data.features.helpers import build_adjusted_data_for_season


def extract_drive_results(raw_plays):
    """Transform play-by-play data into drive-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Drive-level data.
    :rtype: pd.DataFrame
    """
    drives = (raw_plays
              .groupby(['posteam', 'week', 'defteam', 'fixed_drive'])
              .first())
    points = (drives
              ['fixed_drive_result']
              .map({'Touchdown': 7, 'Field goal': 3})
              .fillna(0)
              .groupby(['posteam', 'week', 'defteam'])
              .agg(['sum', 'count']))
    penalties = (drives
                 ['drive_yards_penalized']
                 .groupby(['posteam', 'week', 'defteam'])
                 .agg(['sum', 'count']))
    return points, penalties


def build_drive_features(raw_plays, season):
    """Build features based on drive-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :param int season: Season year.
    :return: Drive-level features.
    :rtype: pd.DataFrame
    """
    features = pd.DataFrame()

    points, penalties = extract_drive_results(raw_plays)
    max_week = points.index.get_level_values('week').max()
    week_nums = range(1, max_week + 1)

    stat_name = 'points_drive'
    season_features = build_adjusted_data_for_season(points, stat_name,
                                                     week_nums, 'mscores')
    features = pd.concat([features, season_features])

    stat_name = 'penalties_drive'
    season_features = build_adjusted_data_for_season(penalties, stat_name,
                                                     week_nums, 'mscores')
    features = pd.concat([features, season_features], axis=1)
    features['season'] = season
    return features


if __name__ == '__main__':
    from src.config.config import PATHS
    from src.data.features.helpers import (calculate_league_averages,
                                           calculate_adversary_mscores,
                                           calculate_adjusted_values,
                                           calculate_adjusted_mscores,
                                           format_mscores)

    raw_plays = pd.read_parquet(PATHS['raw_plays'] / 'play_by_play_2000.parquet')
    raw_plays = raw_plays.query('posteam != "" and posteam != "None"')
    raw_plays = raw_plays.query('week < 7')

    drives_posteam, _ = extract_drive_results(raw_plays)
    print(_)
    _.columns = ['value', 'count']
    drives_defteam = _.swaplevel(0,2).sort_index()

    stat_name = 'points_drive'
    for drives in [_, drives_defteam]:
        obj_side = drives.index.names[0]
        adv_side = drives.index.names[2]

        unstacked = (drives
                     .droplevel(obj_side)
                     .unstack(adv_side))
        league_mean, league_mad = calculate_league_averages(unstacked)
        print(league_mean, league_mad)
        adv_mscores = calculate_adversary_mscores(unstacked, league_mean, league_mad)
        adj_data = calculate_adjusted_values(drives, adv_mscores, league_mean, league_mad)
        adj_mscores = calculate_adjusted_mscores(adj_data, obj_side)
        adj_mscores = format_mscores(adj_mscores, 6, stat_name, obj_side)
        print(adj_mscores)


    # adv_csums = create_weekly_csums(drives, obj_side, adv_side)
    # adv_means = adv_csums['value'] / adv_csums['count']
    # league_mean, league_mad = calculate_league_avgs(adv_csums, adv_means)

    # adv_mscores = calculate_adversary_mscores(adv_means, league_mean,
                                              # league_mad, adv_side)
    # adj_mscores = calculate_adjusted_scores(drives, adv_mscores, league_mean,
                                            # league_mad, adv_side)
    # adj_mscores = format_adj_mscores(adj_mscores, obj_side, stat_name, 5)
    # print(adj_mscores)
