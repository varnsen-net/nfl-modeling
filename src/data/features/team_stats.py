"""Helper functions for building engineered features for team stats."""

import pandas as pd
import numpy as np

from src.utils import map_team_data_to_games
from src.config.config import LOOKBACK_WINDOW, PASSING_AGGS, RUSHING_AGGS


def flatten_columns(df):
    """Flatten the multi-level columns of an aggregation dataframe.
    
    :param df: *pd.DataFrame of shape (n_samples, n_features)*
        Aggregations result with multi-index columns.
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Aggregations with flattened columns.
    """
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def write_cols_to_file(features, output_dir):
    """Write each feature to a separate csv file.
    
    :param features: *pd.DataFrame of shape (n_samples, n_features)*
        Engineered features. Must have columns 'team', 'week', and 'season'.
    :param output_dir: *str*
        Directory to write csv files to.
    :return: *None*
    """
    for col in features.columns:
        if col not in ['team', 'week', 'season']:
            data = features[['season', 'team', 'week', col]]
            data.to_csv(f"{output_dir}/{col}.csv", index=False)
    return


def reframe_team_as_opponent(df):
    """Transform a df index such that 'team' is now 'opponent'.
    
    :param df: *pd.DataFrame of shape (n_samples, n_features)*
        Dataframe with season/team/week/opponent multiindex.
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Dataframe with season/opponent/week multiindex.
    """
    df = df.droplevel('opponent')
    df.index = df.index.rename({'team': 'opponent'})
    df = df.add_prefix('opp_')
    return df


def make_base_points_data(games):
    """Build base points data (e.g. points for and against) for feature
    creation.
    
    :param games: *pd.DataFrame of shape (n_games, n_features)*
        Raw games data.
    :return: *pd.DataFrame of shape (n_samples, 5)*
        Points for/against indexed by season, team, and week.
    """
    home_teams = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']]
    away_teams = games[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']]
    points = pd.DataFrame(np.append(home_teams, away_teams, axis=0))
    points.columns = ['season', 'week', 'team', 'opponent', 'points_for', 'points_against']
    points = (points
              .astype({'points_for': int, 'points_against': int})
              .set_index(['season', 'team', 'week', 'opponent'])
              .sort_index())
    return points


def calculate_avgs(points, window):
    """Calculate averages per team per week per season with expanding or
    rolling windows.
    
    :param points: *pd.DataFrame of shape (n_samples, n_features)*
        Points for/against indexed by season, team, and week.
    :param window: *int*
        Number of weeks to use for rolling averages.
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Points per game averages indexed by season, team, and week.
    """
    group = points.groupby(['season', 'team'])
    if window:
        avgs = (group
                .rolling(window=window)
                .mean()
                .add_suffix(f"_avg_{window}wk"))
    else:
        avgs = (group
                .expanding()
                .mean()
                .add_suffix('_avg'))
    avgs = avgs.droplevel([0,1])
    return avgs


def adjust_for_opponent(base, opponent):
    """Adjust base data for opponent strength.
    
    :param base: *pd.DataFrame of shape (n_samples, n_features)*
        Base data to adjust. Must have season/team/week/opponent multiindex.
    :param opponent: *pd.DataFrame of shape (n_samples, n_features)*
        Opponent data to use for adjustment. Must have season/team/week/opponent
        multiindex.
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Adjusted data.
    """
    opponent = (opponent
                .groupby(['season', 'team'])
                .shift(1)
                .fillna(0))
    opponent = reframe_team_as_opponent(opponent)
    merged = pd.merge(base, opponent, left_index=True, right_index=True)
    base_cols = base.columns
    opp_cols = opponent.columns
    adjusted = merged.loc[:,base_cols] - merged.loc[:,opp_cols].values
    adjusted = adjusted.add_prefix('opp_adj_')
    return adjusted


def make_points_features(games, window, output_dir):
    """Build all points features (e.g. net points).
    
    :param games: *pd.DataFrame of shape (n_games, n_features)*
        Raw games data.
    :param window: *int*
        Number of weeks to use for rolling averages.
    :param output_dir: *str*
        Directory to write csv files to.
    """
    base_points = make_base_points_data(games)
    net_ppg_avgs = calculate_avgs(base_points, window=None)
    net_ppg_avgs = net_ppg_avgs.iloc[:,[1,0]]
    adj_points = adjust_for_opponent(base_points, net_ppg_avgs)
    adj_points['opp_adj_points_net'] = (adj_points['opp_adj_points_for']
                                        - adj_points['opp_adj_points_against'])
    adj_points = calculate_avgs(adj_points, window=None)
    adj_points = (adj_points
                  .round(1)
                  .droplevel('opponent')
                  .swaplevel('week', 'team')
                  .reset_index()
                  .sort_values(['season', 'team', 'week']))
    write_cols_to_file(adj_points, output_dir)
    return


def calculate_squad_aggs(plays, aggregations, squad_type, play_type):
    """Calculate aggregations per team, squad, and week.

    Given a set of play-by-play data:
    - group all plays by team
    - for each group, calculate the aggs for each play using all data up to
      that play
    - now group by team and week and take the last row for each week
    - shift the data down one row so that the stats for each week only go up to
      the end of the previous week
    - format things

    NOTE: aggregations must have 'week':'max' as a key-value pair
    
    :param plays: *pd.DataFrame of shape (n_plays, n_features)*
        Play-by-play data for a given season.
    :param aggregations: *dict*
        Aggregations to calculate for each feature.
    :param squad_type: *str*
        'posteam' or 'defteam'
    :param play_type: *str*
        'pass' or 'rush'
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Aggregated stats for each team and week.
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
    
    :param d_stats: *pd.DataFrame of shape (n_samples, n_features)*
        Aggregated defensive stats returned by calculate_squad_aggs.
    :param stats_to_flip: *list of str*
        List of stat names to search for.
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Defensive stats with flipped signs.
    """
    pattern = '|'.join(stats_to_flip)
    cols_to_flip = d_stats.filter(regex=pattern).columns
    d_stats[cols_to_flip] *= -1
    return d_stats


def assemble_play_type_features(pbp_data, type_aggs_pairs):
    """Assemble features for each play type.
    
    :param pbp_data: *pd.DataFrame of shape (n_plays, n_features)*
        Play-by-play data for a given season.
    :param type_aggs_pairs: *list of tuples*
        List of tuples of the form (play_type, aggregations) where play_type
        is a string and aggregations is a dictionary of aggregations to
        calculate for each feature.
    :return: *pd.DataFrame of shape (n_samples, n_features)*
        Aggregated stats for each team and week.
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


def aggregate_efficiency_stats(games, raw_plays_path, output_dir):
    """Make efficiency stats (e.g. EPA, WPA, etc.) for each team and week.
    
    :param games: *pd.DataFrame of shape (n_games, n_features)*
        Raw games data.
    :param raw_plays_path: *str*
        Path to directory containing raw play-by-play data.
    :param output_dir: *str*
        Path to directory where features will be saved.
    :return: *None*
    """
    game_id_map = pd.DataFrame()
    features = pd.DataFrame()
    for season in list(range(1999, 2024)):
        print(f"Processing season {season}")
        path = f"{raw_plays_path}/play_by_play_{season}.parquet"
        pbp_data = pd.read_parquet(path)
        type_aggs_pairs = [('pass', PASSING_AGGS),
                           ('rush', RUSHING_AGGS),]
        full_stats = assemble_play_type_features(pbp_data, type_aggs_pairs)
        full_stats['season'] = season
        features = pd.concat([features, full_stats])
        id_map = pbp_data.groupby(['game_id']).agg({'away_team': 'first',
                                                    'home_team': 'first',
                                                    'week': 'first'})
        id_map = id_map.reset_index()
        id_map['season'] = season
        game_id_map = pd.concat([game_id_map, id_map])
    write_cols_to_file(features, output_dir)
    return


def build_team_stats_features(raw_games_path, raw_plays_path, output_dir):
    """Build engineered features for team stats.
    
    :param raw_games_path: *str*
        Path to directory containing raw games data.
    :param raw_plays_path: *str*
        Path to directory containing raw play-by-play data.
    :param output_dir: *str*
        Path to directory where features will be saved.
    :return: *None*
    """
    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    make_points_features(games, LOOKBACK_WINDOW, output_dir)
    aggregate_efficiency_stats(games, raw_plays_path, output_dir)
    return


if __name__ == '__main__':
    from src.config.config import PATHS

    games = pd.read_csv(PATHS['raw_games']).dropna(subset=['result'])
