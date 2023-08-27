import csv

import numpy as np
import pandas as pd
import requests


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
    weather = weather.set_index('game_id')
    weather.loc[indoor_games.index, weather_cols.keys()] = weather_cols.values()
    weather = weather.reset_index()
    return weather


def make_weather_features(games, weather, weather_feat_path):
    """Make weather features.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame weather: raw weather data
    :param str weather_feat_path: path to save weather features
    :return: None
    :rtype: None
    """
    weather = fix_indoor_weather(games, weather)
    weather.to_csv(f"{weather_feat_path}/weather.csv", index=False)
    return


