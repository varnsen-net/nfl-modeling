"""Helper functions for building engineered features for game-level statistics."""

import pandas as pd
import numpy as np

from src.data.features.helpers import build_adjusted_features

def extract_points_game(raw_plays):
    """Extract the total points per game for each team.

    Example output:

    .. code-block:: none

                                     points_game  count
        posteam season week defteam                    
        ARI     2022   1    KC              21.0      1
                       2    LV              23.0      1
                       3    LA               9.0      1
                       4    CAR             26.0      1
                       5    PHI             17.0      1
        ...                                  ...    ...
        WAS     2024   7    CAR             40.0      1
                       8    CHI             18.0      1
                       9    NYG             27.0      1
                       10   PIT             27.0      1
                       11   PHI             16.0      1

    
    :param pd.DataFrame raw_plays: The raw play-by-play data.
    :return: Total points per team per game.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .groupby(['posteam', 'season', 'week', 'defteam'])
            .last()
            ['posteam_score']
            .to_frame("points_game")
            .assign(count=1))


def extract_normal_points_game(normal_plays):
    """Extract the total points per game for each team for normal-game plays.
    
    Normal-game implies that score and time are (presumably) not a factor in
    decision-making.

    Example output:

    .. code-block:: none

                                     points_normal_game  count
        posteam season week defteam                           
        ARI     2022   1    KC                     14.0      1
                       2    LV                     21.0      1
                       3    LA                      9.0      1
                       4    CAR                    27.0      1
                       5    PHI                    17.0      1
        ...                                         ...    ...
        WAS     2024   7    CAR                    30.0      1
                       8    CHI                    12.0      1
                       9    NYG                    27.0      1
                       10   PIT                    27.0      1
                       11   PHI                    17.0      1
    
    :param pd.DataFrame normal_plays: Play-by-play data reduced to normal-game
        plays.
    :return: Total points per team per game.
    :rtype: pd.DataFrame
    """
    return (normal_plays
            .groupby(['posteam', 'season', 'week', 'defteam',
                      'fixed_drive'])
            .first()
            ['fixed_drive_result']
            .map({'Touchdown': 7, 'Field goal': 3})
            .fillna(0)
            .groupby(['posteam', 'season', 'week', 'defteam'])
            .sum()
            .to_frame('points_normal_game')
            .assign(count=1))


def extract_home_away_points_game(raw_plays):
    """Extract the total points per game for each team, separated by home/away.

    This function unstacks before computing rolling sums because simply
    grouping by team will make it so we can't shift the week numbers correctly
    later in the pipeline.

    Example output:

    .. code-block:: none

                               points_for  points_against
        season home_team week                            
        2022   ARI       1           21.0            44.0
                         2           21.0            44.0
                         3           33.0            64.0
                         4           33.0            64.0
                         5           50.0            84.0
        ...                           ...             ...
        2024   WAS       7          115.0           103.0
                         8          133.0           118.0
                         9          133.0           118.0
                         10         160.0           146.0
                         11         150.0           119.0
    
    :param pd.DataFrame raw_plays: The raw play-by-play data.
    :return: Total points per team per game, separated by home/away.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    for team_type in ['home', 'away']:
        opp_type = 'away' if team_type == 'home' else 'home'
        yield (raw_plays
               .groupby([f'{team_type}_team', 'season', 'week'])
               [[f'total_{team_type}_score', f'total_{opp_type}_score']]
               .last()
               .rename(columns={f'total_{team_type}_score': 'points_for',
                                f'total_{opp_type}_score': 'points_against'})
               .unstack(f'{team_type}_team')
               .rolling(16, min_periods=1)
               .sum()
               .stack(f'{team_type}_team', future_stack=True)
               .reorder_levels([0, 2, 1])
               .sort_index())


def calculate_pythagorean_expectation(scores, x=2.77):
    """Calculate the Pythagorean expectation for a set of rolling scores.

    Example input:

    .. code-block:: none

                               points_for  points_against
        season home_team week                            
        2022   ARI       1           21.0            44.0
                         2           21.0            44.0
                         3           33.0            64.0
                         4           33.0            64.0
                         5           50.0            84.0
        ...                           ...             ...
        2024   WAS       7          115.0           103.0
                         8          133.0           118.0
                         9          133.0           118.0
                         10         160.0           146.0
                         11         150.0           119.0

    Example output:

    .. code-block:: none

        season  home_team  week
        2022    ARI        1       0.114166
                           2       0.114166
                           3       0.137670
                           4       0.137670
                           5       0.192001
                                     ...   
        2024    WAS        7       0.575729
                           8       0.582117
                           9       0.582117
                           10      0.563073
                           11      0.655044
    
    :param pd.DataFrame scores: Rolling scores for a team.
    :param float x: The exponent for the Pythagorean expectation.
    :return: The Pythagorean expectations.
    :rtype: pd.Series
    """
    points_for = scores['points_for']
    points_against = scores['points_against']
    return points_for ** x / (points_for ** x + points_against ** x)


def build_pythagorean_features(raw_plays):
    """Build Pythagorean expectation features for each season/team/week.

    Example output:

    .. code-block:: none

                          pythag_exp_h  pythag_exp_a
        season team week                            
        2022   ARI  1         0.114166           NaN
                    2         0.114166      0.655226
                    3         0.137670      0.655226
                    4         0.137670      0.721567
                    5         0.192001      0.721567
        ...                        ...           ...
        2024   WAS  7         0.375633      0.382692
                    8         0.397684      0.376958
                    9         0.433543      0.404964
                    10        0.439710      0.514129
                    11        0.563073      0.487296
    
    :param pd.DataFrame raw_plays: The raw play-by-play data.
    :return: The Pythagorean expectation features.
    :rtype: pd.DataFrame
    """
    home_points, away_points = extract_home_away_points_game(raw_plays)
    home_pythag = calculate_pythagorean_expectation(home_points)
    away_pythag = calculate_pythagorean_expectation(away_points)
    pythag_features = pd.concat([home_pythag, away_pythag], axis=1)
    pythag_features.columns = ['pythag_exp_h', 'pythag_exp_a']
    pythag_features.index.names = ['season', 'team', 'week']
    return pythag_features


def build_game_features(raw_plays, normal_plays):
    """Build all points features (e.g. net points).

    Output example:

    .. code-block:: none

                          adj_points_game_pos  adj_points_game_def  ...  pythag_exp_a
        season team week                                            ...              
        2022   ARI  1                     NaN                  NaN  ...           NaN
                    2               -3.500000             0.285714  ...      0.655226
                    3               -5.870968             0.000000  ...      0.655226
                    4               -3.570552            -0.145985  ...      0.721567
                    5               -3.033113            -0.090909  ...      0.721567
        ...                               ...                  ...  ...           ...
        2024   WAS  7                0.256961             0.334211  ...      0.382692
                    8               -0.041163            -0.021308  ...      0.376958
                    9               -0.025737            -0.057623  ...      0.404964
                    10               0.003510             0.089762  ...      0.514129
                    11              -0.227493             0.123421  ...      0.487296
    
    :param pd.DataFrame raw_plays: The raw play-by-play data.
    :param pd.DataFrame normal_plays: The normal-game play-by-play data.
    :return: The engineered features.
    :rtype: pd.DataFrame
    """
    all_points = extract_points_game(raw_plays)
    adj_points = build_adjusted_features(all_points)

    normal_points = extract_normal_points_game(normal_plays)
    adj_normal_points = build_adjusted_features(normal_points)

    pythag_exp = build_pythagorean_features(raw_plays)
    return pd.concat([adj_points, adj_normal_points, pythag_exp], axis=1)


if __name__ == '__main__':
    from src.data.build import preprocess_plays, reduce_to_normal_plays
    from src.config.config import PATHS

    raw_plays_path = PATHS['raw_plays']

    seasons = list(range(2022, 2025))
    raw_plays = pd.concat([pd.read_parquet(raw_plays_path / f"play_by_play_{season}.parquet")
                           for season in seasons])
    pythag_features = build_pythagorean_features(raw_plays)

    processed_plays = preprocess_plays(raw_plays)
    normal_plays = reduce_to_normal_plays(processed_plays)
    points_features = build_game_features(processed_plays, normal_plays)
    print(points_features)

