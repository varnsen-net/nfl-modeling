""""""

import pandas as pd

from src.config.config import CURRENT_SEASON
from src.data.features.helpers import build_adjusted_data_for_season


def extract_play_results(raw_plays):
    """Transform play-by-play data into play-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Series-level data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'week', 'defteam'])
            ['yards_gained']
            .agg(['sum', 'count']))


def build_play_features(raw_plays, season):
    """Build features based on play-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :param int season: Season to build features for.
    :return: Features.
    :rtype: pd.DataFrame
    """
    features = pd.DataFrame()

    yards = extract_play_results(raw_plays)
    max_week = yards.index.get_level_values('week').max()
    week_nums = range(1, max_week + 1)

    stat_name = 'yards_drive'
    season_features = build_adjusted_data_for_season(yards, stat_name,
                                                     week_nums, 'mscores')
    season_features['season'] = season
    features = pd.concat([features, season_features])
    return features



if __name__ == '__main__':
    from src.config.config import PATHS
    features = build_play_features(PATHS['raw_plays'])
    print(features)

