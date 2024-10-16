""""""

import pandas as pd

from src.config.config import CURRENT_SEASON


def create_weekly_csums(raw_data, obj_side, adv_side):
    """Create cumulative sums for the adversary side of the ball for the raw
    data.

    Takes in a dataframe that looks like this:

                              value  count
    posteam week defteam              
    ARI     1    WAS        9.0     12
            2    NYG       27.0     10
            3    DAL       30.0      9
            4    SF        17.0      7
    ATL     1    CAR       24.0     13
    ...                     ...    ...
    TEN     4    CIN       27.0      9
    WAS     1    ARI       20.0     13
            2    DEN       34.0     11
            3    BUF        3.0     11
            4    PHI       31.0     10

    And returns a dataframe that looks like this:

                 value               ...    count                  
    defteam    ARI   ATL   BAL   BUF ...   SF    TB   TEN   WAS
    week                             ...                       
    1         20.0  10.0   9.0  16.0 ... 11.0  11.0  12.0  12.0
    2         51.0  34.0  26.0  26.0 ... 21.0  21.0  23.0  23.0
    3         67.0  54.0  48.0  29.0 ... 31.0  31.0  32.0  34.0
    4        102.0  70.0  51.0  50.0 ... 38.0  42.0  41.0  44.0

    :param pd.DataFrame raw_data: Raw data.
    :param str obj_side: Side of the ball that is the object of the analysis.
    :param str adv_side: Side of the ball that is the adversary.
    :return: Cumulative sums of the adversary side of the ball.
    :rtype: pd.DataFrame
    """
    return (raw_data
            .droplevel(obj_side)
            .unstack(adv_side)
            .expanding()
            .sum())


def calculate_league_avgs(adv_csums, adv_means):
    """Calculate the league average and median absolute deviation.

    Takes in data that looks like this:

             value                ... count                              
    posteam    ARI    ATL    BAL  ...    TB   TEN   WAS
    week                          ...                  
    1         27.0   10.0   20.0  ...   9.0  13.0   9.0
    2         68.0   33.0   43.0  ...  20.0  24.0  17.0
    3         81.0   50.0   78.0  ...  31.0  35.0  25.0
    4         95.0   69.0  113.0  ...  42.0  48.0  34.0

    And:

    posteam       ARI       ATL       BAL  ...        TB       TEN       WAS
    week                                   ...                              
    1        2.700000  1.000000  2.000000  ...  4.111111  1.307692  2.333333
    2        3.578947  1.736842  1.954545  ...  2.850000  1.416667  2.470588
    3        2.700000  1.724138  2.294118  ...  2.064516  1.371429  3.200000
    4        2.435897  1.769231  2.511111  ...  2.333333  1.604167  3.558824
    
    :param pd.DataFrame adv_csums: Cumulative sums of the adversary side of
        the ball.
    :param pd.DataFrame adv_means: Mean values of the adversary side of the
        ball.
    :return: League average and median absolute deviation.
    :rtype: Tuple[float, float]
    """
    values_total = adv_csums['value'].iloc[-1,:].sum()
    counts_total = adv_csums['count'].iloc[-1,:].sum()
    league_mean = values_total / counts_total
    league_mad = (adv_means.iloc[-1,:] - league_mean).abs().median()
    return league_mean, league_mad


def calculate_adversary_mscores(adv_means, league_mean, league_mad, adv_side):
    """Calculate the adversary m-scores based on league mean and median
        absolute deviation.
    
    Takes in data that looks like this:

    posteam       ARI       ATL       BAL  ...        TB       TEN       WAS
    week                                   ...                              
    1        2.700000  1.000000  2.000000  ...  4.111111  1.307692  2.333333
    2        3.578947  1.736842  1.954545  ...  2.850000  1.416667  2.470588
    3        2.700000  1.724138  2.294118  ...  2.064516  1.371429  3.200000
    4        2.435897  1.769231  2.511111  ...  2.333333  1.604167  3.558824

    Returns data that looks like this:

    posteam  week
    ARI      1       0.000000
             2       2.011162
             3       4.421989
             4       2.011162
                       ...   
    WAS      1       0.000000
             2       1.005448
             3       1.381918
             4       3.382591
    
    :param pd.DataFrame adv_means: Mean values of the adversary side of the
        ball.
    :param float league_mean: League average.
    :param float league_mad: Median absolute deviation.
    :param str adv_side: Side of the ball that the adversary is on.
    :return: Adversary m-scores.
    :rtype: pd.Series
    """
    adv_mscores = (adv_means - league_mean) / league_mad
    adv_mscores = (adv_mscores
                   .shift(1)
                   .fillna(0)
                   .stack()
                   .reorder_levels([adv_side, 'week'])
                   .sort_index()
                   .rename('adv_mscore'))
    return adv_mscores


def calculate_adjusted_scores(raw_data, adv_mscores, league_mean, league_mad,
                              adv_side):
    """Convert raw values and counts into league and opponent-adjusted scores.

    Takes in data that looks like this:

                          value  count
    posteam week defteam              
    ARI     1    BUF       27.0     10
            2    LA        41.0      9
            3    DET       13.0     11
            4    WAS       14.0      9
    ...                     ...    ...
    WAS     1    TB        21.0      9
            2    NYG       21.0      8
            3    CIN       38.0      8
            4    ARI       41.0      9

    And adversary m-scores that look like this:

    posteam  week
    ARI      1       0.000000
             2       2.011162
             3       4.421989
             4       2.011162
                       ...   
    WAS      1       0.000000
             2       1.005448
             3       1.381918
             4       3.382591

    And returns data that looks like this:

    posteam  week  defteam
    ARI      1     BUF        2.011162
             2     LA         5.089524
             3     DET       -6.574983
             4     WAS       -3.139048
                                ...   
    WAS      1     TB         1.005448
             2     NYG        0.800000
             3     CIN        6.252101
             4     ARI        3.718095

    :param pd.DataFrame raw_data: Raw data. Should have columns 'value' and
        'count'.
    :param pd.Series adv_mscores: Adversary m-scores.
    :param float league_mean: League average.
    :param float league_mad: Median absolute deviation.
    :param str adv_side: Side of the ball that the adversary is on.
    :return: Adjusted scores.
    :rtype: pd.Series
    """
    adj_data = raw_data.join(adv_mscores, on=[adv_side, 'week'], how='left')
    raw_rate = adj_data['value'] / adj_data['count']
    mscores = (raw_rate - league_mean) / league_mad
    adj_mscores = mscores - adj_data['adv_mscore']
    return adj_mscores


def format_adj_mscores(adj_mscores, obj_side, stat_name, week):
    """Format adjusted m-scores for a given side of the ball.

    Takes in data that looks like this:

    posteam  week  defteam
    ARI      1     BUF        2.011162
             2     LA         5.089524
             3     DET       -6.574983
             4     WAS       -3.139048
                                ...   
    WAS      1     TB         1.005448
             2     NYG        0.800000
             3     CIN        6.252101
             4     ARI        3.718095

    And returns data that looks like this:

                  points_drive_def
    defteam week                  
    ARI     5            -0.124675
    ATL     5            -0.457409
    BAL     5             1.795325
                            ...
    TB      5             1.005448
    TEN     5             0.000000
    WAS     5             5.881638

    :param pd.Series adj_mscores: Adjusted m-scores.
    :param str obj_side: Side of the ball that is the object of the analysis.
    :param str stat_name: Name of the statistic.
    :param int week: Week number to assign to the data
    :return: Formatted adjusted m-scores.
    :rtype: pd.Series
    """
    return (adj_mscores
            .groupby(obj_side)
            .median()
            .rename(f'{stat_name}_{obj_side[:3]}')
            .to_frame()
            .assign(week=week)
            .set_index('week', append=True))


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
            adv_csums = create_weekly_csums(df, obj_side, adv_side)
            adv_means = adv_csums['value'] / adv_csums['count']
            league_mean, league_mad = calculate_league_avgs(adv_csums, adv_means)

            adv_mscores = calculate_adversary_mscores(adv_means, league_mean,
                                                      league_mad, adv_side)
            adj_mscores = calculate_adjusted_scores(df, adv_mscores, league_mean,
                                                    league_mad, adv_side)
            adj_mscores = format_adj_mscores(adj_mscores, obj_side, stat_name, w)
            feature_df = pd.concat([feature_df, adj_mscores])
        feature_df = feature_df.sort_index()
        season_df = season_df.join(feature_df, how='left')
    season_df[f"{stat_name}_def"] = season_df[f"{stat_name}_def"] * -1
    season_df[f"{stat_name}_net"] = (season_df[f"{stat_name}_pos"]
                                         + season_df[f"{stat_name}_def"])
    season_df = season_df.droplevel('defteam')
    season_df.index.names = ['team', 'week']
    return season_df


