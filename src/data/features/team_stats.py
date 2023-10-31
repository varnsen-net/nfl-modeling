"""Helper functions for building engineered features for team stats."""

import pandas as pd
import numpy as np

from src.utils import map_team_data_to_games
from src.config.config import PASSING_AGGS, RUSHING_AGGS


def flatten_columns(df):
    """Flatten multi-index columns.
    
    :param pd.DataFrame df: dataframe with multi-index columns
    :return: dataframe with flattened columns
    :rtype: pd.DataFrame
    """
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def calculate_cumulative_points(games):
    """Calculate the cumulative points for/against each team up to each week.
    
    :param pd.DataFrame games: raw games dataframe
    :return: cumulative points for/against each team *up to* the indexed week
    :rtype: pd.DataFrame
    """
    home_teams = games[['season', 'week', 'home_team', 'home_score', 'away_score']]
    away_teams = games[['season', 'week', 'away_team', 'away_score', 'home_score']]
    points = pd.DataFrame(np.append(home_teams, away_teams, axis=0))
    points.columns = ['season', 'week', 'team', 'points_for', 'points_against']
    points = (points
              .astype({'points_for': int, 'points_against': int})
              .set_index(['season', 'team', 'week'])
              .sort_index())
    points['cpf'] = points.groupby(['season', 'team']).cumsum()['points_for']
    points['cpa'] = points.groupby(['season', 'team']).cumsum()['points_against']
    return points


def calculate_pythag_exp(points, x=2.68):
    """Calculates the pythagorean expectation for a set of points for and
    points against.
    
    :param pd.DataFrame points: df with points for and points against columns
    :param float x: exponent to use in pythagorean expectation formula
    :return: pythagorean expectations
    :rtype: pd.Series
    """
    numerator = points['cpf']**x
    denominator = points['cpf']**x + points['cpa']**x
    pythag_exp = numerator / denominator
    return pythag_exp


def make_pythag_exp_feature(games, feature_name):
    """Build pythagorean expectation feature.
    
    :param pd.DataFrame games: raw games dataframe
    :param str feature_name: name of feature to create
    :return: None
    :rtype: None
    """
    cumulative_points = calculate_cumulative_points(games)
    pythag_exp = calculate_pythag_exp(cumulative_points)
    pythag_exp.name = feature_name
    return pythag_exp


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
    
    :param pd.DataFrame plays: raw play-by-play data
    :param dict aggregations: aggregations to calculate
    :param str squad_type: side to group by. must be 'posteam' or 'defteam'
    :param str play_type: play type to prefix column names with
    :return: aggregated stats. each row is a team and week number. columns are
        features
    :rtype: pd.DataFrame
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


def assemble_play_type_features(pbp_data, type_aggs_pairs):
    """Assemble features for each play type.
    
    :param pd.DataFrame pbp_data: raw play-by-play data
    :param list type_aggs_pairs: list of tuples. each tuple is a play type and
        a dict of aggregations to calculate
    :return: aggregated stats. each row is a team and week number. columns are
        features
    :rtype: pd.DataFrame
    """
    full_stats = pd.DataFrame(columns=['team', 'week'])
    for play_type, aggregations in type_aggs_pairs:
        plays = pbp_data[pbp_data[play_type] == 1]
        o_stats = calculate_squad_aggs(plays, aggregations,
                                       'posteam', play_type)
        d_stats = calculate_squad_aggs(plays, aggregations,
                                       'defteam', play_type)
        merged = o_stats.merge(d_stats, on=['team', 'week'])
        full_stats = full_stats.merge(merged, on=['team', 'week'],
                                      how='outer')
    return full_stats


def make_team_stats_features(games, raw_plays_path, output_dir):
    """Make team stats features.
    
    :param pd.DataFrame games: game data
    :param str raw_plays_path: path to raw play-by-play data
    :param str output_dir: path to output directory
    :return: None
    :rtype: None
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
    for col in features.columns:
        if col not in ['team', 'week', 'season']:
            data = features[['season', 'team', 'week', col]]
            data.to_csv(f"{output_dir}/{col}.csv", index=False)
    return


def build_features(raw_games_path, raw_plays_path, output_dir,
                   **kwargs):
    """Build engineered features for team stats.
    
    :param str raw_games_path: path to raw games data
    :param str raw_plays_path: path to raw play-by-play data
    :param str output_dir: path to save stats features
    :param dict kwargs: additional arguments
    :return: None
    :rtype: None
    """
    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    pythag_exp = make_pythag_exp_feature(games, 'pythagorean_expectation')
    pythag_exp.to_csv(f"{output_dir}/pythagorean_expectation.csv")
    make_team_stats_features(games, raw_plays_path, output_dir)
    return
