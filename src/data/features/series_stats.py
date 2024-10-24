""""""

import pandas as pd

from src.data.features.helpers import build_adjusted_data_for_season


def extract_series_results(raw_plays):
    """Transform play-by-play data into series-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Series-level data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'week', 'defteam', 'series'])
            .first()
            ['series_success']
            .groupby(['posteam', 'week', 'defteam'])
            .agg(['sum', 'count']))


def build_series_features(raw_plays, season):
    """Build features based on series-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :param int season: Season.
    :return: Series-level features.
    :rtype: pd.DataFrame
    """
    series = extract_series_results(raw_plays)
    max_week = series.index.get_level_values('week').max()
    week_nums = range(1, max_week + 1)

    stat_name = 'series_success'
    season_features = build_adjusted_data_for_season(series, stat_name,
                                                     week_nums, 'mscores')
    season_features['season'] = season
    return season_features



if __name__ == '__main__':
    from src.config.config import PATHS
    features = build_series_features(PATHS['raw_plays'])
    print(features)

