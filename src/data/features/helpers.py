""""""

import pandas as pd

from src.config.config import CURRENT_SEASON


def calc_cumulative_means(df, side):
    """Calculate the cumulative means for the teams in the given dataframe.
    
    :param pd.DataFrame df: A dataframe with a multiindex of posteam, week,
        and defteam, and columns sum and count.
    :param str side: The side of the ball to calculate the cumulative means.
        Must be either 'posteam' or 'defteam'.
    :return: A dataframe with the cumulative means for each team.
    :rtype: pd.DataFrame
    """
    return (df
            .groupby(side)
            .cumsum()
            .assign(mean=lambda x: x['sum'] / x['count']))


def shift_one_week_forward(df, side):
    """Shift the cumulative means one week forward.

    :param pd.DataFrame df: A dataframe with a multiindex of posteam and week.
    :param str side: The side of the ball to shift the cumulative means.
        Must be either 'posteam' or 'defteam'.
    :return: A dataframe with the cumulative means shifted one week forward.
    :rtype: pd.DataFrame
    """
    return (df
            .groupby(side)
            .shift(1)
            .fillna(0))


def get_most_recent_means(df, side):
    """Get the most recent cumulative means for each team.
    
    :param pd.DataFrame df: A dataframe with a multiindex of posteam and week.
    :param str side: The side of the ball to calculate the cumulative means.
        Must be either 'posteam' or 'defteam'.
    :return: A dataframe with the most recent cumulative means for each team.
    :rtype: pd.DataFrame
    """
    return (df
            .groupby(side)
            .last())


def calc_adversary_raw_means(df, side):
    """Calculate the raw cumulative means for the opponents of the teams in
    the given dataframe. Will be used to adjust weekly means.

    Suppose we have a df that counts points per drive for each week.
    The input df will look like this:
    
                           sum  count
    posteam week defteam             
    ARI     1    BUF      27.0     10
            2    LA       41.0      9
            3    DET      13.0     11
    ATL     1    PIT      10.0     10
            2    PHI      23.0      9
    ...                    ...    ...
    TEN     2    NYJ      17.0     11
            3    GB       14.0     11
    WAS     1    TB       21.0      9
            2    NYG      21.0      8
            3    CIN      38.0      8
    
    Running this function with side='defteam' will return the raw cumulative
    mean for each team on the defensive side of the ball up to the most recent
    week. The output will look like this:

    defteam
    ARI    2.650000
    ATL    2.179487
    BAL    2.045455
    BUF    2.000000
    CAR    2.866667
    ...
    SEA    1.620000
    SF     2.000000
    TB     1.925000
    TEN    1.500000
    WAS    3.312500

    :param pd.DataFrame df: A dataframe with a multiindex of posteam, week,
        and defteam, and columns sum and count.
    :param str side: The side of the ball to calculate the cumulative means.
        Must be either 'posteam' or 'defteam'.
    :return: A dataframe with the most recent cumulative means for each team.
    :rtype: pd.DataFrame
    """
    return (df
            .sort_index(level=[side, 'week'])
            .pipe(calc_cumulative_means, side)
            .pipe(shift_one_week_forward, side)
            .pipe(get_most_recent_means, side)
            ['mean'])


def calc_adjusted_means(df, adv_means, obj_side, adv_side, stat_name):
    """Calculate the cumulative adjusted means for the teams in the given
    dataframe.

    The input dataframe should have a multiindex of posteam, week, and defteam,
    and columns sum, count, and game mean. For example:
    
                           sum  count  game_mean
    posteam week defteam
    ARI     1    BUF      27.0     10   2.700000
            2    LA       41.0      9   4.555556
            3    DET      13.0     11   1.181818

    The input adv_means is a series with the most recent cumulative means for
    the opponents of the teams in the given dataframe. For example:

    defteam
    ARI    2.650000
    ATL    2.179487
    BAL    2.045455
    BUF    2.000000
    CAR    2.866667
    ...
    SEA    1.620000
    SF     2.000000
    TB     1.925000
    TEN    1.500000
    WAS    3.312500

    The output will have a multiindex of posteam, week, and defteam, and columns
    for the adjusted means for the statistic of interest. For example:

               stat_mean_pos
    team week               
    ARI  1          2.700000
         2          0.995215
         3          0.364921
         4         -0.088132
    ATL  1          1.000000
    ...                  ...
    TEN  4         -0.132440
    WAS  1          2.333333
         2          0.336898
         3          1.250632
         4          1.426777

    :param pd.DataFrame df: A dataframe with a multiindex of posteam, week,
        and defteam, and columns sum, count, and game mean.
    :param pd.Series adv_means: A series with the most recent cumulative means
        for the opponents of the teams in the given dataframe.
    :param str obj_side: The side of the ball to make adjustments on.
    :param str adv_side: The side of the ball to use for adjustments.
    :param str stat_name: The name of the statistic to calculate adjusted means.
    :return: A dataframe with the adjusted means for the statistic of interest.
    :rtype: pd.DataFrame
    """
    return (df
            .join(adv_means, on=adv_side)
            .rename(columns={'mean': 'opp_mean'})
            .assign(resid_mean=lambda x: x['game_mean'] - x['opp_mean'])
            .assign(adj_sum=lambda x: x['count'] * x['resid_mean'])
            [['count', 'adj_sum']]
            .groupby(obj_side)
            .cumsum()
            .assign(adj_mean=lambda x: x['adj_sum'] / x['count'])
            .rename(columns={'adj_mean': f"{stat_name}_mean_{obj_side[:3]}"}))


def build_adjusted_data_for_season(raw_df, stat_name, week_nums):
    """Build a dataframe with adjusted means for some statistic for each team
    and game week in a season.
    
    The input data should have a multiindex of posteam, week, and defteam, and
    columns sum and count. For example:
    
                           sum  count
    posteam week defteam             
    ARI     1    BUF      27.0     10
            2    LA       41.0      9
            3    DET      13.0     11
    ATL     1    PIT      10.0     10
            2    PHI      23.0      9
    ...                    ...    ...
    TEN     2    NYJ      17.0     11
            3    GB       14.0     11
    WAS     1    TB       21.0      9
            2    NYG      21.0      8
            3    CIN      38.0      8

    The output will have a multiindex of team and week, and columns for the
    adjusted means for the statistic of interest. For example:
    
               stat_mean_pos  stat_mean_def  stat_mean_net
    team week                                             
    ARI  1          2.700000       2.700000       0.000000
         2          0.995215       1.855556      -0.860340
         3          0.364921      -2.397129       2.762050
         4         -0.088132      -1.144444       1.056313
    ATL  1          1.000000       1.000000       0.000000
    ...                  ...            ...            ...
    TEN  4         -0.132440       0.033952      -0.166392
    WAS  1          2.333333       2.333333       0.000000
         2          0.336898       0.122807       0.214091
         3          1.250632       0.513273       0.737358
         4          1.426777       0.050449       1.376328

    :param pd.DataFrame raw_df: A dataframe with a multiindex of posteam, week,
        and defteam, and columns sum and count.
    :param str stat_name: The name of the statistic to calculate adjusted means.
    :param list week_nums: A list of the weeks in the season to calculate the
        adjusted means.
    :return: A dataframe with the adjusted means for the statistic of interest.
    :rtype: pd.DataFrame
    """
    season_df = pd.DataFrame(index=raw_df.index)
    for obj_side in ['posteam', 'defteam']:
        adv_side = 'defteam' if obj_side == 'posteam' else 'posteam'
        feature_df = pd.DataFrame()
        for w in week_nums:
            df = raw_df.query('week <= @w').copy()
            df['game_mean'] = df['sum'] / df['count']
            adv_means = calc_adversary_raw_means(df, adv_side)
            df = calc_adjusted_means(df, adv_means, obj_side, adv_side,
                                     stat_name)
            df = (df
                  .iloc[:, [-1]]
                  .query('week == @w'))
            feature_df = pd.concat([feature_df, df])
        feature_df = feature_df.sort_index()
        season_df = season_df.join(feature_df, how='outer')
    season_df[f"{stat_name}_mean_def"] = season_df[f"{stat_name}_mean_def"] * -1
    season_df[f"{stat_name}_mean_net"] = (season_df[f"{stat_name}_mean_pos"]
                                         + season_df[f"{stat_name}_mean_def"])
    season_df = season_df.droplevel('defteam')
    season_df.index.names = ['team', 'week']
    return season_df


