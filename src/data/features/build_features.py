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
    games = pd.read_csv(f"{RAW_GAMES_PATH}/games.csv")
    games = preprocess_games_data(games)
    city_coords = pd.read_csv(f"{CITY_COORDS_PATH}/city-coordinates.csv")
    weather = pd.read_csv(f"{RAW_WEATHER_PATH}/weather.csv")

    make_travel_features(games, city_coords)
    make_stats_features(games)
    make_weather_features(games, weather)
