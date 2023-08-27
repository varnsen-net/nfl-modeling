import json

import pandas as pd
import numpy as np

from src.data.features.travel import make_travel_features
from src.data.features.stats import make_stats_features
from src.data.features.weather import make_weather_features


with open('config.json') as f:
    config = json.load(f)
    RAW_GAMES_PATH = config['paths']['raw_games']
    RAW_WEATHER_PATH = config['paths']['raw_weather']
    CITY_COORDS_PATH = config['paths']['city_coords']
    STATS_FEAT_PATH = config['paths']['stats_features']
    TRAVEL_FEAT_PATH = config['paths']['travel_features']
    WEATHER_FEAT_PATH = config['paths']['weather_features']


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


if __name__ == '__main__':
    games = pd.read_csv(RAW_GAMES_PATH)
    games = preprocess_games_data(games)
    city_coords = pd.read_csv(CITY_COORDS_PATH)
    weather = pd.read_csv(RAW_WEATHER_PATH)

    make_travel_features(games, city_coords, TRAVEL_FEAT_PATH)
    make_stats_features(games, STATS_FEAT_PATH)
    make_weather_features(games, weather, WEATHER_FEAT_PATH)
