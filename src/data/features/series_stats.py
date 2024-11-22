""""""

import pandas as pd

from src.data.features.helpers import build_adjusted_features


def extract_series_results(raw_plays):
    """Transform play-by-play data into series-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Series-level data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'season', 'week', 'defteam', 'series'])
            .first()
            ['series_success']
            .groupby(['posteam', 'season', 'week', 'defteam'])
            .agg(['sum', 'count']))


def build_series_features(raw_plays):
    """Build features based on series-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Series-level features.
    :rtype: pd.DataFrame
    """
    series = extract_series_results(raw_plays)
    series.columns = ['success_series', 'count']
    adj_series = build_adjusted_features(series)
    return adj_series
