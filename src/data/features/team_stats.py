"""Helper functions for building engineered features for team stats."""

import pandas as pd
import numpy as np

from src.config.config import CURRENT_SEASON, PASSING_AGGS, RUSHING_AGGS


def reduce_plays(plays):
    """Reduce play by play data to 'normal' game situations where playcalling
    is less effected by game state.

    :param plays: Play-by-play data.
    :type plays: pd.DataFrame of shape (n_plays, n_features)
    :return: Reduced play-by-play data.
    :rtype: pd.DataFrame of shape (n_plays, n_features)
    """
    plays = (plays
             .loc[plays['half_seconds_remaining'] > 120]
             .loc[plays['score_differential'].between(-16, 16)])
    return plays


def flatten_columns(df):
    """Flatten the multi-level columns of an aggregation dataframe.
    
    :param df: Aggregations result with multi-index columns.
    :type df: pd.DataFrame of shape (n_samples, n_features)
    :return: Aggregations with flattened columns.
    :rtype: pd.DataFrame of shape (n_samples, n_features)
    """
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def write_cols_to_file(features, output_dir):
    """Write each feature to a separate csv file.
    
    :param features: Engineered features. Must have columns 'team', 'week', and 'season'.
    :type features: pd.DataFrame of shape (n_samples, n_features)
    :param str output_dir: Directory to write csv files to.
    :return: None
    :rtype: None
    """
    for col in features.columns:
        if col not in ['team', 'week', 'season']:
            data = features[['season', 'team', 'week', col]]
            data.to_csv(f"{output_dir}/{col}.csv", index=False)
    return


def calculate_squad_aggs(plays, aggregations, squad_type, play_type):
    """Calculate aggregations per team, squad, and week.

    Given a set of play-by-play data:
    - group all plays by team

    - for each group, calculate the aggs for each play using all data up to that play

    - now group by team and week and take the last row for each week

    - shift the data down one row so that the stats for each week only go up to the end of the previous week

    - format things

    NOTE: aggregations must have 'week':'max' as a key-value pair
    
    :param plays: Play-by-play data.
    :type plays: pd.DataFrame of shape (n_plays, n_features)
    :param dict aggregations: Aggregations to calculate.
    :param str squad_type: Squad type to aggregate by (e.g. posteam, defteam).
    :param str play_type: Play type to aggregate by (e.g. pass, rush).
    :return: Aggregated data.
    :rtype: pd.DataFrame of shape (n_samples, n_features)
    """
    side = ('o' if squad_type == 'posteam' else
            'd' if squad_type == 'defteam' else
            None)
    aggregated = (plays
                  .groupby([squad_type])
                  .expanding()
                  .agg(aggregations)
                  .pipe(flatten_columns)
                  .groupby([squad_type, 'week_max'])
                  .last()
                  .add_prefix(f'{play_type}_')
                  .add_prefix(f'{side}_')
                  .reset_index()
                  .rename(columns={'week_max': 'week',
                                   squad_type: 'team'})
                  .astype({'week': 'int'}))
    return aggregated


def flip_defense_stat_signs(d_stats, stats_to_flip):
    """Flip the sign on certain defensive stats (e.g. EPA) for which
    negative values are good.

    Use regex to find columns with names that contain any of the strings in
    stats_to_flip. Multiply those columns by -1.
    
    :param d_stats: Aggregated defensive stats returned by calculate_squad_aggs.
    :type d_stats: pd.DataFrame of shape (n_samples, n_features)
    :param list[str] stats_to_flip: Strings to use for regex matching. Each
        element should take the form '_statname_'.
    :return: Defensive stats with signs flipped.
    :rtype: pd.DataFrame of shape (n_samples, n_features)
    """
    pattern = '|'.join(stats_to_flip)
    cols_to_flip = d_stats.filter(regex=pattern).columns
    d_stats[cols_to_flip] *= -1
    return d_stats


def assemble_play_type_features(pbp_data, type_aggs_pairs):
    """Assemble features for each play type.
    
    :param pbp_data: Play-by-play data for a given season.
    :type pbp_data: pd.DataFrame of shape (n_plays, n_features)
    :param list[tuple] type_aggs_pairs: List of tuples of the form
        (play_type, aggregations) where play_type is a string and aggregations
        is a dictionary of aggregations to calculate for each feature.
    :return: Aggregated data for each team and week.
    :rtype: pd.DataFrame of shape (n_samples, n_features)
    """
    full_stats = pd.DataFrame(columns=['team', 'week'])
    for play_type, aggregations in type_aggs_pairs:
        plays = pbp_data[pbp_data[play_type] == 1]
        o_stats = calculate_squad_aggs(plays, aggregations,
                                       'posteam', play_type)
        d_stats = calculate_squad_aggs(plays, aggregations,
                                       'defteam', play_type)
        d_stats = flip_defense_stat_signs(d_stats, ['_epa_', '_wpa_'])
        merged = o_stats.merge(d_stats, on=['team', 'week'])
        full_stats = full_stats.merge(merged, on=['team', 'week'],
                                      how='outer')
    return full_stats


def build_team_efficiency_features(raw_plays_path):
    """Make efficiency stats (e.g. EPA, WPA, etc.) for each team and week.
    
    :param str raw_plays_path: Path to raw play-by-play data.
    :return: None
    :rtype: None
    """
    features = pd.DataFrame()
    for season in list(range(1999, CURRENT_SEASON + 1)):
        print(f"Processing season {season}")
        path = f"{raw_plays_path}/play_by_play_{season}.parquet"
        pbp_data = pd.read_parquet(path)
        pbp_data = reduce_plays(pbp_data)
        type_aggs_pairs = [('pass', PASSING_AGGS),
                           ('rush', RUSHING_AGGS),]
        full_stats = assemble_play_type_features(pbp_data, type_aggs_pairs)
        full_stats['season'] = season
        features = pd.concat([features, full_stats])
    features = features.set_index(['season', 'team', 'week']).sort_index()
    return features
