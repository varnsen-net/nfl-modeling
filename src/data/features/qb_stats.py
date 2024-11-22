import pandas as pd

from src.config.config import PATHS


def preprocess_plays(raw_plays):
    """Preprocess raw play-by-play data.

    :param pd.DataFrame raw_plays: Raw play-by-play data.
    :return: Preprocessed play-by-play data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .query('posteam != "" and posteam != "None"')
            .query('half_seconds_remaining > 120')
            .query('score_differential.abs() <= 16')
            .query('pass_attempt == 1'))


if __name__ == '__main__':
    raw_plays_path = PATHS['raw_plays']

    seasons = list(range(2001, 2025))
    plays = pd.concat([pd.read_parquet(raw_plays_path / f"play_by_play_{season}.parquet")
                           for season in seasons])
    pass_plays = preprocess_plays(plays)
    print(pass_plays)

    aggs = {'qb_epa': 'sum',
            'pass_attempt': 'count'}

    qb_stats = (pass_plays
                .groupby(['passer_player_name', 'season', 'week'])
                .agg(aggs)
                .groupby('passer_player_name')
                .rolling(32, min_periods=1)
                .sum()
                .assign(qb_epa_per_play=lambda x: x['qb_epa'] / x['pass_attempt'])
                .droplevel(0))
    print(qb_stats)

    starters = (pass_plays
                .groupby(['season', 'posteam', 'week', 'defteam'])
                ['passer_player_name']
                .first()
                .to_frame('passer_player_name')
                .set_index('passer_player_name', append=True)
                .merge(qb_stats, how='left', left_index=True, right_index=True)
                .reset_index()
                [['posteam', 'week', 'qb_epa_per_play', 'season']]
                .rename(columns={'posteam': 'team'}))
    print(starters)
    starters.to_csv(PATHS['features'] / 'qb_features.csv', index=False)
