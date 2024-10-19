"""Helper functions for building engineered features for team points."""

import pandas as pd
import numpy as np

from src.data.features.helpers import build_adjusted_data_for_season

def extract_game_results(raw_plays):
    """"""
    points = (raw_plays
              .groupby(['posteam', 'week', 'defteam'])
              .last()
              ['posteam_score']
              .to_frame("value")
              .assign(count=1))
    return points


def calculate_adj_pythag_exp(adj_points):
    """"""
    points_for_sq = adj_points['adj_points_game_pos'] ** 2.68
    points_against_sq = adj_points['adj_points_game_def'] ** 2.68
    adj_pythag_exp = points_for_sq / (points_for_sq + points_against_sq)
    adj_pythag_exp = adj_pythag_exp.to_frame('adj_pythag_exp')
    return adj_pythag_exp


def build_game_features(raw_plays, season):
    """Build all points features (e.g. net points).
    
    """
    features = pd.DataFrame()
    points = extract_game_results(raw_plays)
    max_week = points.index.get_level_values('week').max()
    week_nums = range(1, max_week + 1)

    stat_name = 'points_game'
    season_features = build_adjusted_data_for_season(points,
                                                     stat_name,
                                                     week_nums,
                                                     'mscores')
    features = pd.concat([features, season_features])

    # no stat name needed here -- will be renamed by pythag_exp fcn
    season_features = build_adjusted_data_for_season(points,
                                                     stat_name,
                                                     week_nums,
                                                     'values')
    adj_pythag_exp = calculate_adj_pythag_exp(season_features)
    features = pd.concat([features, adj_pythag_exp], axis=1)
    features['season'] = season
    return features


if __name__ == '__main__':
    from src.config.config import PATHS

    raw_plays = pd.read_parquet(PATHS['raw_plays'] / 'play_by_play_2024.parquet')
    raw_plays = raw_plays.query('posteam != "" and posteam != "None"')
    raw_plays = raw_plays.query('week < 7')

    features = build_game_features(raw_plays, 2000)
    print(features)
    print(features.query('team == "KC"'))
