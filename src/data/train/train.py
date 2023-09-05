import json

import numpy as np
import pandas as pd


with open('config.json') as f:
    config = json.load(f)
    RAW_GAMES_PATH = config['paths']['raw_games']
    STATS_FEAT_PATH = config['paths']['stats_features']
    TRAVEL_FEAT_PATH = config['paths']['travel_features']
    WEATHER_FEAT_PATH = config['paths']['weather_features']
    TRAINING_DATA_PATH = config['paths']['training_data']


def preprocess_games_data(games):
    """Reduce the games dataframe -- only consider regular season games played
    at home.
    
    :param pd.DataFrame games: raw games dataframe
    :return: reduced games data
    :rtype: pd.DataFrame
    """
    games = (games
             .loc[games['game_type'] == 'REG']
             .loc[games['location'] == 'Home']
             .drop(columns=['game_type', 'location'])
             .dropna(subset=['result'])
             .copy())
    return games


def merge_pythag_exp(games, pythag_exp):
    """Merge pythagorean expectation data with games data.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame pythag_exp: pythagorean expectation data
    :return: games data with pythagorean expectation data
    :rtype: pd.DataFrame
    """
    games = (games
             .merge(pythag_exp,
                    left_on=['season', 'away_team', 'week'],
                    right_on=['season', 'team', 'week'],
                    how='inner')
             .drop(columns=['team'])
             .merge(pythag_exp,
                    left_on=['season', 'home_team', 'week'],
                    right_on=['season', 'team', 'week'],
                    how='inner', suffixes=('_away', '_home'))
             .drop(columns=['team']))
    return games


def merge_travel(games, travel):
    """Merge travel data with games data.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame travel: travel data
    :return: games data with travel data
    :rtype: pd.DataFrame
    """
    games = games.merge(travel, on=['game_id'], how='inner')
    return games


def merge_weather(games, weather):
    """Merge weather data with games data.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame weather: weather data
    :return: games data with weather data
    :rtype: pd.DataFrame
    """
    games = games.merge(weather, on=['game_id'], how='inner')
    return games


def make_target_col(games):
    """Make target column for games data.
    
    :param pd.DataFrame games: games data
    :return: games data with target column
    :rtype: pd.DataFrame
    """
    games['target'] = np.where(games['result'] > 0, 1, 0)
    games = games.drop(columns=['result'])
    return games


if __name__ == '__main__':
    games = pd.read_csv(RAW_GAMES_PATH)
    games = games[['game_id', 'season', 'week', 'away_team', 'home_team',
                   'away_rest', 'home_rest', 'div_game', 'surface',
                   'result']]

    pythag_exp = pd.read_csv(STATS_FEAT_PATH)
    games = merge_pythag_exp(games, pythag_exp)

    travel = pd.read_csv(TRAVEL_FEAT_PATH)
    games = merge_travel(games, travel)

    weather = pd.read_csv(WEATHER_FEAT_PATH)
    games = merge_weather(games, weather)

    games = make_target_col(games)
    games.to_csv(TRAINING_DATA_PATH, index=False)
    games.to_html('tmp.html')
