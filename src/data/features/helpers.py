""""""

import pandas as pd


def calc_cumulative_stats(raw_features, obj_side, opp_side, window=None):
    """Calculate the cumulative stats for all opponents.

    :param pd.DataFrame raw_features: The raw features to calculate the
        cumulative stats for.
    :param str obj_side: The side of the ball we're building adjusted features
        for.
    :param str opp_side: The side of the ball we're getting cumulative stats
        for.
    :param int window: The window size to use for rolling stats. If None, then
        use expanding stats.
    :return: The cumulative stats for all opponents.
    :rtype: pd.DataFrame
    """
    if window:
        return (raw_features
                .droplevel(obj_side)
                .unstack(opp_side)
                .rolling(window=window, min_periods=1)
                .sum())
    else:
        return (raw_features
                .droplevel(obj_side)
                .unstack(opp_side)
                .groupby('season')
                .expanding()
                .sum()
                .droplevel(0))


def build_adjusted_features(raw_features):
    """"""
    raw_features_def = (raw_features
                        .reset_index()
                        .set_index(['defteam', 'season', 'week', 'posteam'])
                        .sort_index())
    adjusted_features = pd.DataFrame()
    for df in [raw_features, raw_features_def]:
        obj_side = df.index.names[0]
        opp_side = df.index.names[3]

        opp_rolling_stats = calc_cumulative_stats(df, obj_side, opp_side,
                                                  window=16)

        opp_rolling_counts = opp_rolling_stats.pop('count')
        opp_rolling_means = (opp_rolling_stats
                             .div(opp_rolling_counts, axis=0)
                             # .groupby('season')
                             .shift(1)
                             .ffill())

        weekly_values = (df
                          .droplevel(opp_side)
                          .unstack(obj_side))
        game_counts = weekly_values.pop('count')

        opp_expected_results = opp_rolling_means.mul(game_counts, level=opp_side)

        adj_game_results = weekly_values - opp_expected_results

        adj_rolling_stats = (adj_game_results
                             .groupby('season')
                             .expanding()
                             .sum()
                             .droplevel(0))

        stat_names = adj_rolling_stats.columns.get_level_values(0).unique()
        cols = {name: adj_rolling_stats[name].abs().median(axis=1)
                for name in stat_names}
        league_median_abs_devs = pd.DataFrame(cols)


        for c in league_median_abs_devs.columns:
            adj_rolling_stats[c] = adj_rolling_stats[c].div(league_median_abs_devs[c],
                                                            axis=0)
        adj_rolling_stats = (adj_rolling_stats
                             .stack(obj_side, future_stack=True)
                             .reset_index()
                             .rename(columns={obj_side: 'team'})
                             .set_index(['season', 'team', 'week'])
                             .sort_index()
                             .add_prefix('adj_')
                             .add_suffix(f'_{obj_side[:3]}'))
        adjusted_features = pd.concat([adjusted_features, adj_rolling_stats],
                                      axis=1)
    return adjusted_features


