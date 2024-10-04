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


def build_play_features(raw_plays_path):
    """Build features based on play-level data.

    :param pathlib.Path raw_plays_path: Path to raw play-by-play data.
    :return: Series-level features.
    :rtype: pd.DataFrame
    """
    features = pd.DataFrame()
    stat_name = 'ypp'
    for season in list(range(1999, CURRENT_SEASON + 1)):
        print(f"Processing season {season}")
        raw_plays = raw_plays_path / f"play_by_play_{season}.parquet"
        raw_plays = pd.read_parquet(raw_plays)
        query = "play_type == 'pass' or play_type == 'run'"
        raw_plays = raw_plays.query(query)
        plays = extract_play_results(raw_plays)
        max_week = plays.index.get_level_values('week').max()
        week_nums = range(1, max_week + 1)
        season_features = build_adjusted_data_for_season(plays,
                                                         stat_name,
                                                         week_nums)
        season_features['season'] = season
        features = pd.concat([features, season_features])
    return features



if __name__ == '__main__':
    from src.config.config import PATHS
    features = build_play_features(PATHS['raw_plays'])
    print(features)

