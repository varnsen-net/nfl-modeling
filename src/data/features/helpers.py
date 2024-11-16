""""""

import pandas as pd


def calculate_league_averages(data):
    """Calculate the league average and median absolute deviation for a
    statistic.
    
    Takes in data that looks like this:

            value              ... count                                    
    defteam   ARI   ATL   BAL  ...    TB   TEN   WAS
    week                       ...                  
    1        34.0  18.0  27.0  ...   9.0  13.0   9.0
    2        10.0  20.0  26.0  ...  11.0  11.0   7.0
    3        21.0  23.0  27.0  ...  10.0  11.0   7.0
    4        41.0  24.0  10.0  ...  10.0  13.0   9.0
    5        16.0  30.0  35.0  ...  11.0   NaN  13.0
    6        34.0  20.0  23.0  ...  15.0  10.0   9.0

    Returns the league average and median absolute deviation.
    
    :param pd.DataFrame data: Raw data. Should have columns 'value' and
        'count'.
    :return: League average and median absolute deviation.
    :rtype: tuple[float, float]
    """
    team_vals = data['value'].sum()
    team_counts = data['count'].sum()
    team_means = team_vals / team_counts
    league_mean = team_vals.sum() / team_counts.sum()
    league_mad = (team_means - league_mean).abs().median()
    return league_mean, league_mad


def calculate_adversary_mscores(data, league_mean, league_mad):
    """Calculate the adversary m-scores based on league mean and median
    absolute deviation.
    
    Takes in data that looks like this:

            value              ... count                                    
    defteam   ARI   ATL   BAL  ...    TB   TEN   WAS
    week                       ...                  
    1        34.0  18.0  27.0  ...   9.0  13.0   9.0
    2        10.0  20.0  26.0  ...  11.0  11.0   7.0
    3        21.0  23.0  27.0  ...  10.0  11.0   7.0
    4        41.0  24.0  10.0  ...  10.0  13.0   9.0
    5        16.0  30.0  35.0  ...  11.0   NaN  13.0
    6        34.0  20.0  23.0  ...  15.0  10.0   9.0

    Returns data that looks like this:

                  adv_mscore
    week defteam            
    1    ARI        1.297050
         ATL        1.282418
         BAL        0.527680
         BUF       -0.508345
         CAR        2.501187
    ...                  ...
    6    SEA       -1.167418
         SF         0.185210
         TB         0.775601
         TEN       -1.668693
         WAS        2.135715

    :param pd.DataFrame data: Raw data. Should have columns 'value' and
        'count'.
    :param float league_mean: League average.
    :param float league_mad: Median absolute deviation.
    :return: Adversary m-scores.
    :rtype: pd.DataFrame
    """
    adv_mscores = pd.DataFrame()
    max_week = data.index.get_level_values('week').max()
    for w in range(1, max_week + 1):
        unstacked_minus_week = data.query('week != @w')
        team_vals = unstacked_minus_week['value'].sum()
        team_counts = unstacked_minus_week['count'].sum()
        team_means = team_vals / team_counts
        team_mscores = (team_means - league_mean) / league_mad
        team_mscores = team_mscores.rename("adv_mscore")
        team_mscores = pd.concat([team_mscores], keys=[w], names=['week'])
        adv_mscores = pd.concat([adv_mscores, team_mscores.to_frame()])
    return adv_mscores


def calculate_adjusted_values(data, adv_mscores, league_mean, league_mad):
    """Convert raw values and counts into league and opponent-adjusted values.

    Takes in data that looks like this:

                          value  count
    posteam week defteam              
    ARI     1    BUF       27.0     10
            2    LA        41.0      9
            3    DET       13.0     11
            4    WAS       14.0      9
            5    SF        23.0     10
    ...                     ...    ...
    WAS     2    NYG       21.0      8
            3    CIN       38.0      8
            4    ARI       41.0      9
            5    CLE       34.0     12
            6    BAL       23.0      9

    And adversary m-scores that look like this:

                  adv_mscore
    week defteam            
    1    ARI        1.297050
         ATL        1.282418
         BAL        0.527680
         BUF       -0.508345
         CAR        2.501187
    ...                  ...
    6    SEA       -1.167418
         SF         0.185210
         TB         0.775601
         TEN       -1.668693
         WAS        2.135715

    And returns data that looks like this:

                          adj_value  count
    posteam week defteam                  
    ARI     1    BUF      28.529207     10
            2    LA       37.286088      9
            3    DET      15.324075     11
            4    WAS       5.017795      9
            5    SF       22.464217     10
    ...                         ...    ...
    WAS     2    NYG      25.085994      8
            3    CIN      38.306727      8
            4    ARI      39.113949      9
            5    CLE      36.623727     12
            6    BAL      21.280953      9

    :param pd.DataFrame data: Raw data. Should have columns 'value' and
        'count'.
    :param pd.Series adv_mscores: Adversary m-scores.
    :param float league_mean: League average.
    :param float league_mad: Median absolute deviation.
    :return: Adjusted scores with counts.
    :rtype: pd.DataFrame
    """
    adj_data = data.join(adv_mscores, how='left')
    adj_data['game_avg'] = adj_data['value'] / adj_data['count']
    adj_data['game_mscore'] = (adj_data['game_avg'] - league_mean) / league_mad
    adj_data['adj_mscore'] = adj_data['game_mscore'] - adj_data['adv_mscore']
    adj_data['adj_avg'] = adj_data['adj_mscore'] * league_mad + league_mean
    adj_data['adj_value'] = adj_data['adj_avg'] * adj_data['count']
    return adj_data[['adj_value', 'count']]


def calculate_median_mscores(adj_data, obj_side):
    """Calculate the adjusted m-scores for a team's performance.
    
    Takes in data that looks like this:

                          adj_value  count
    posteam week defteam                  
    ARI     1    BUF      28.529207     10
            2    LA       37.286088      9
            3    DET      15.324075     11
            4    WAS       5.017795      9
            5    SF       22.464217     10
    ...                         ...    ...
    WAS     2    NYG      25.085994      8
            3    CIN      38.306727      8
            4    ARI      39.113949      9
            5    CLE      36.623727     12
            6    BAL      21.280953      9

    And returns data that looks like this:

             adj_mscore
    posteam            
    ARI        0.204046
    ATL        0.786494
    BAL        1.420507
                    ...
    TB         1.743112
    TEN       -0.367255
    WAS        3.084125

    :param pd.DataFrame adj_data: Adjusted data. Should have columns 'adj_value'
        and 'count'.
    :param str obj_side: The side of the ball that the team is on. Should be
        either 'posteam' or 'defteam'.
    :return: Adjusted m-scores.
    :rtype: pd.DataFrame
    """
    totals = adj_data.groupby(obj_side).sum()
    means = totals['adj_value'] / totals['count']
    league_mean_adj = means.mean()
    league_mad_adj = (means - league_mean_adj).abs().median()
    median_mscores = (means - league_mean_adj) / league_mad_adj
    return median_mscores.to_frame('adj_mscore')


def calculate_value_sums(adj_data, obj_side):
    """"""
    return (adj_data
            ['adj_value']
            .groupby(obj_side)
            .sum()
            .to_frame('adj_sum'))


def format_adj_scores(data, week_num, stat_name, obj_side):
    """Format the adjusted m-scores for a team's performance.
    
    Takes in data that looks like this:

             adj_mscore
    posteam            
    ARI        0.204046
    ATL        0.786494
    BAL        1.420507
                    ...
    TB         1.743112
    TEN       -0.367255
    WAS        3.084125

    And returns data that looks like this:

                  adj_points_drive_pos
    posteam week                      
    ARI     6                 0.204046
    ATL     6                 0.786494
    BAL     6                 1.420507
                                   ...
    TB      6                 1.743112
    TEN     6                -0.367255
    WAS     6                 3.084125

    :param pd.DataFrame data: Adjusted data.
    :param int week_num: The week number.
    :param str stat_name: The name of the statistic.
    :param str obj_side: The side of the ball that the team is on. Should be
        either 'posteam' or 'defteam'.
    :return: Formatted m-scores.
    :rtype: pd.DataFrame
    """
    col_name = data.columns[0]
    return (data
            .assign(week=week_num)
            .set_index('week', append=True)
            .sort_index()
            .rename(columns={col_name: f"adj_{stat_name}_{obj_side[:3]}"}))


def build_adjusted_data_for_season(raw_df, stat_name, week_nums, output_type):
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
    raw_df.columns = ['value', 'count']
    raw_df_defteam = (raw_df
                      .reset_index()
                      .set_index(['defteam', 'week', 'posteam'])
                      .sort_index())

    for df in [raw_df, raw_df_defteam]:
        obj_side = df.index.names[0]
        adv_side = df.index.names[2]
        feature_df = pd.DataFrame()
        for w in week_nums:
            reduced = raw_df.query('week <= @w').copy()
            unstacked = (reduced
                         .droplevel(obj_side)
                         .unstack(adv_side))
            league_mean, league_mad = calculate_league_averages(unstacked)
            adv_mscores = calculate_adversary_mscores(unstacked, league_mean,
                                                      league_mad)
            adj_data = calculate_adjusted_values(reduced, adv_mscores,
                                                 league_mean, league_mad)
            if output_type == 'values':
                adj_value_sums = calculate_value_sums(adj_data, obj_side)
                adj_value_sums = format_adj_scores(adj_value_sums, w,
                                                   stat_name, obj_side)
                feature_df = pd.concat([feature_df, adj_value_sums])
            if output_type == 'mscores':
                median_mscores = calculate_median_mscores(adj_data, obj_side)
                median_mscores = format_adj_scores(median_mscores, w,
                                                   stat_name, obj_side)
                feature_df = pd.concat([feature_df, median_mscores])
        feature_df = feature_df.sort_index()
        season_df = season_df.join(feature_df, how='left')
    season_df = season_df.droplevel('defteam')
    season_df.index.names = ['team', 'week']
    return season_df


