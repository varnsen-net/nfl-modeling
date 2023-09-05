import json
import csv

import numpy as np
import pandas as pd
import requests

from src.utils import parse_common_args


def fix_indoor_weather(games, weather):
    """Fix the weather data for indoor games.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame weather: raw weather data
    :return: weather data with indoor games fixed
    :rtype: pd.DataFrame
    """
    weather_cols = {'temp_C': 21.0,
                    'feelslike_C': 21.0,
                    'rel_humidity_pc': 50,
                    'dewpoint_C': 16.0,
                    'wind_speed_kmh': 0.0,
                    'cloudcover_pc': 100,
                    'rain_mm': 0.0,
                    'snowfall_cm': 0.0}
    indoor_labels = ['dome', 'closed']
    indoor_games = games.query('roof in @indoor_labels').set_index('game_id')
    indoor_weather = pd.DataFrame(weather_cols, index=indoor_games.index)
    weather = weather.set_index('game_id')
    weather.update(indoor_weather)
    weather = weather.reset_index()
    return weather


def make_weather_features(games, weather, weather_metadata, output_path):
    """Make weather features.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame weather: raw weather data
    :param dict weather_metadata: weather features metadata
    :param str output_path: path to save weather features
    :return: None
    :rtype: None
    """
    weather = fix_indoor_weather(games, weather)
    for feature in weather_metadata.values():
        col = feature['api_name']
        feature_data = weather[['game_id', col]]
        feature_data.to_csv(f"{output_path}/{col}.csv", index=False)
    return


if __name__ == '__main__':
    args = parse_common_args()
    config_path = args.c
    raw_games_path = args.g
    raw_weather_path = args.w
    output_path = args.o

    with open(config_path) as f:
        config = json.load(f)
        weather_metadata = config['features']['weather']

    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    weather = pd.read_csv(raw_weather_path)
    make_weather_features(games, weather, weather_metadata, output_path)
