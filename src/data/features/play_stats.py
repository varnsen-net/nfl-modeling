""""""

import pandas as pd

from src.data.features.helpers import build_adjusted_features


def extract_play_results(raw_plays):
    """Transform play-by-play data into play-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Series-level data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'season', 'week', 'defteam'])
            .agg({'epa': 'sum',
                  'qb_epa': 'sum',
                  'play_id': 'count'})
            .sort_index())


def format_col_names(col_names):
    """"""
    return [f"{col}_play" if col != 'play_id'
            else 'count'
            for col in col_names]


def build_play_features(raw_plays):
    """Build features based on play-level data.

    :param pd.DataFrame raw_plays: Play-by-play data.
    :return: Features.
    :rtype: pd.DataFrame
    """
    run_pass_plays = raw_plays.query('pass_attempt == 1 or rush_attempt == 1')
    raw_features = extract_play_results(run_pass_plays)
    raw_features.columns = format_col_names(raw_features.columns)
    adjusted_features = build_adjusted_features(raw_features)
    return adjusted_features




if __name__ == '__main__':
    from src.config.config import PATHS

    raw_plays_path = PATHS['raw_plays']

    seasons = list(range(2021, 2025))
    plays = pd.concat([pd.read_parquet(raw_plays_path / f"play_by_play_{season}.parquet")
                           for season in seasons])
    play_features = build_play_features(plays)
    print(play_features)

