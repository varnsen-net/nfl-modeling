""""""

import polars as pl


def make_rolling_data(game_data, stat_name, side):
    """Turn weekly data into a set of rolling data that we can use to make
    adjustments to the game data.

    Example game_data input:

    | season | week | posteam | defteam | points_drive | count |
    | ---    | ---  | ---     | ---     | ---          | ---   |
    | i32    | i32  | str     | str     | i64          | u32   |
    |--------|------|---------|---------|--------------|-------|
    | 2012   | 16   | BUF     | MIA     | 10           | 11    |
    | 2004   | 3    | NYG     | CLE     | 27           | 10    |
    | 2013   | 12   | CAR     | MIA     | 20           | 13    |
    | 2012   | 2    | BUF     | KC      | 28           | 14    |
    | 2011   | 1    | BAL     | PIT     | 34           | 14    |
    | ...    | ...  | ...     | ...     | ...          | ...   |
    | 2001   | 2    | CIN     | BAL     | 14           | 13    |
    | 2010   | 14   | ATL     | CAR     | 31           | 13    |
    | 2019   | 16   | HOU     | TB      | 16           | 14    |
    | 2022   | 16   | GB      | MIA     | 26           | 11    |
    | 2023   | 15   | WAS     | LA      | 21           | 12    |

    There are three rolling data outputs:

    rolled_data_obj: Holds all of the rolling window information for the
    object teams.

    Example:

    | posteam | season | index | week | defteam                   | points_drive     | count            |
    | ---     | ---    | ---   | ---  | ---                       | ---              | ---              |
    | str     | i32    | u32   | i32  | list[str]                 | list[i64]        | list[u32]        |
    |---------|--------|-------|------|---------------------------|------------------|------------------|
    | ARI     | 2001   | 0     | 2    | ["DEN"]                   | [20]             | [12]             |
    | ARI     | 2001   | 1     | 3    | ["DEN", "ATL"]            | [20, 14]         | [12, 11]         |
    | ARI     | 2001   | 2     | 4    | ["DEN", "ATL", "PHI"]     | [20, 14, 14]     | [12, 11, 12]     |
    | ARI     | 2001   | 3     | 5    | ["DEN", "ATL", ... "CHI"] | [20, 14, ... 13] | [12, 11, ... 11] |
    | ARI     | 2001   | 4     | 6    | ["DEN", "ATL", ... "KC"]  | [20, 14, ... 24] | [12, 11, ... 12] |
    | ...     | ...    | ...   | ...  | ...                       | ...              | ...              |
    | WAS     | 2024   | 12275 | 13   | ["TB", "NYG", ... "TEN"]  | [21, 21, ... 42] | [9, 8, ... 12]   |
    | WAS     | 2024   | 12276 | 15   | ["TB", "NYG", ... "NO"]   | [21, 21, ... 20] | [9, 8, ... 9]    |
    | WAS     | 2024   | 12277 | 16   | ["TB", "NYG", ... "PHI"]  | [21, 21, ... 35] | [9, 8, ... 14]   |
    | WAS     | 2024   | 12278 | 17   | ["TB", "NYG", ... "ATL"]  | [21, 21, ... 31] | [9, 8, ... 10]   |
    | WAS     | 2024   | 12279 | 18   | ["TB", "NYG", ... "DAL"]  | [21, 21, ... 24] | [9, 8, ... 10]   |

    opp_agged_data: Holds the rolling totals for the opponent teams.

    Example:

    | defteam | season | index | week | points_drive | count |
    | ---     | ---    | ---   | ---  | ---          | ---   |
    | str     | i32    | u32   | i32  | i64          | u32   |
    |---------|--------|-------|------|--------------|-------|
    | ARI     | 2001   | 0     | 2    | 37           | 12    |
    | ARI     | 2001   | 1     | 3    | 71           | 23    |
    | ARI     | 2001   | 2     | 4    | 91           | 35    |
    | ARI     | 2001   | 3     | 5    | 104          | 46    |
    | ARI     | 2001   | 4     | 6    | 120          | 58    |
    | ...     | ...    | ...   | ...  | ...          | ...   |
    | WAS     | 2024   | 12275 | 13   | 310          | 134   |
    | WAS     | 2024   | 12276 | 15   | 330          | 143   |
    | WAS     | 2024   | 12277 | 16   | 363          | 157   |
    | WAS     | 2024   | 12278 | 17   | 387          | 166   |
    | WAS     | 2024   | 12279 | 18   | 406          | 177   |

    league_means: Holds the rolling totals for the league.

    Example:

    | season | week | points_drive | count |
    | ---    | ---  | ---          | ---   |
    | i32    | i32  | i64          | u32   |
    |--------|------|--------------|-------|
    | 2001   | 1    | 482          | 367   |
    | 2001   | 2    | 982          | 700   |
    | 2001   | 3    | 1557         | 1040  |
    | 2001   | 4    | 2062         | 1402  |
    | 2001   | 5    | 2649         | 1733  |
    | ...    | ...  | ...          | ...   |
    | 2024   | 14   | 9052         | 4454  |
    | 2024   | 15   | 9760         | 4805  |
    | 2024   | 16   | 10506        | 5152  |
    | 2024   | 17   | 11213        | 5485  |
    | 2024   | 18   | 11962        | 5831  |

    :param pl.DataFrame game_data: The game data DataFrame.
    :param str stat_name: The name of the statistic to be adjusted.
    :param str side: The side of the ball (posteam or defteam).
    :return: The rolling data for the object teams, opponent teams, and
        league means.
    :rtype: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
    """
    period = '20i'
    opp_side = 'defteam' if side == 'posteam' else 'posteam'

    # Collect rolling data for the object teams
    game_data_obj = (
        game_data
        .sort(side, 'season', 'week')
        .with_row_index('index')
    )
    rolled_data_obj = (
        game_data_obj.rolling(
            index_column='index',
            group_by=[side, 'season'],
            period=period,
        )
        .agg(
            pl.col('week').last(),
            pl.col(opp_side),
            pl.col(stat_name, 'count')
        )
    )

    # Collect rolling data for the opponent teams
    game_data_opp = (
        game_data
        .sort(opp_side, 'season', 'week')
        .with_row_index('index')
    )
    opp_agged_data = (
        game_data_opp.rolling(
            index_column='index',
            group_by=[opp_side, 'season'],
            period=period,
        )
        .agg(
            pl.col('week').last(),
            pl.col(stat_name, 'count').sum()
        )
    )

    # Collect rolling data for the league
    league_means = (
        game_data
        .group_by(['season', 'week'])
        .agg(
            pl.col(stat_name).sum(),
            pl.col('count').sum()
        )
        .sort('season', 'week')
        .rolling(
            index_column='week',
            period=period,
            group_by=['season']
        )
        .agg(
            pl.col(stat_name).sum(),
            pl.col('count').sum()
        )
    )
    return rolled_data_obj, opp_agged_data, league_means


def calculate_avg_adj_metric(rolled_data, opp_agged_data, league_means,
                             stat_name, side):
    """Summary.

    Basic steps:
    1. Explode the rolled object team columns that hold lists and join them back up.
    2. Join that to the opponent and league data.
    3. Calculate everything
    """
    opp_side = 'defteam' if side == 'posteam' else 'posteam'
    exploded_opps = rolled_data.explode(opp_side).select(side, 'season', 'week', opp_side)
    exploded_stat = rolled_data.explode(stat_name).select('season', 'week', stat_name)
    exploded_count = rolled_data.explode('count').select('season', 'week', 'count')
    return (
        pl.concat(
            [exploded_opps,
             exploded_stat.select(stat_name),
             exploded_count.select('count')],
            how='horizontal',
        )
        .join(
            opp_agged_data,
            on=[opp_side, 'season', 'week'],
            how='left',
            suffix=f'_{opp_side}'
        )
        .join(
            league_means,
            on=['season', 'week'],
            how='left',
            suffix='_lg'
        )
        .with_columns(
            (pl.col(f"{stat_name}_{opp_side}") - pl.col(stat_name)).alias(f"{stat_name}_{opp_side}_resid"),
            (pl.col(f"count_{opp_side}") - pl.col('count')).alias(f"count_{opp_side}_resid"),
            (pl.col(f"{stat_name}_lg") - pl.col(f"{stat_name}_{opp_side}")).alias(f"{stat_name}_lg_resid"),
            (pl.col(f"count_lg") - pl.col(f"count_{opp_side}")).alias(f"count_lg_resid"),
        )
        .with_columns(
            game_avg=pl.col(stat_name) / pl.col('count'),
            opp_avg=pl.col(f"{stat_name}_{opp_side}_resid") / pl.col(f"count_{opp_side}_resid"),
            lg_avg=pl.col(f"{stat_name}_lg_resid") / pl.col(f"count_lg_resid"),
        )
        .with_columns(
            adj_score=pl.col(stat_name) * (pl.col('lg_avg') / pl.col('opp_avg')),
        )
        .group_by([side, 'season', 'week'])
        .agg(
            pl.col('adj_score').sum().alias('adj_score_total'),
            pl.col('count').sum().alias('count_total'),
        )
        .with_columns((pl.col('adj_score_total') / pl.col('count_total')).alias('avg_adj_score'))
        .select(
            pl.col(side).alias('team'),
            'season', 'week',
            pl.col('avg_adj_score').alias(f"{stat_name}_scaled_{side}"),
        )
    )


def build_adjusted_features(feature_data):
    """"""
    stat_name = feature_data.collect_schema().names()[-2]
    dfs_to_join = []
    for side in ['posteam', 'defteam']:
        rolled_data, opp_agged_data, league_means = make_rolling_data(
            feature_data,
            stat_name,
            side,
        )
        adjusted_data = calculate_avg_adj_metric(
            rolled_data,
            opp_agged_data,
            league_means,
            stat_name,
            side,
        )
        dfs_to_join.append(adjusted_data)
    left_df, right_df = dfs_to_join
    final_df = (
        left_df
        .join(right_df,
              on=['team', 'season', 'week'],
              how='left')
        .sort('team', 'season', 'week')
    )
    return final_df
