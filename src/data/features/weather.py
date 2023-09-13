import json
import csv

import numpy as np
import pandas as pd
import requests


def fix_indoor_weather(games, weather, defaults):
    """Fix the weather data for indoor games.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame weather: raw weather data
    :param dict defaults: default weather values
    :return: weather data with indoor games fixed
    :rtype: pd.DataFrame
    """
    indoor_labels = ['dome', 'closed']
    indoor_games = games.query('roof in @indoor_labels').set_index('game_id')
    indoor_weather = pd.DataFrame(defaults, index=indoor_games.index)
    weather = weather.set_index('game_id')
    weather.update(indoor_weather)
    weather = weather.reset_index()
    return weather


def build_features(config_path, raw_games_path, raw_weather_path, output_dir, **kwargs):
    """Make weather features.
    
    :param str config_path: path to config file
    :param str raw_games_path: path to raw games data
    :param str raw_weather_path: path to raw weather data
    :param str output_dir: path to output directory
    :param dict kwargs: keyword arguments
    :return: None
    :rtype: None
    """
    with open(config_path) as f:
        config = json.load(f)
        weather_metadata = config['features']['weather']
    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    weather = pd.read_csv(raw_weather_path)
    metadata_values = weather_metadata.values()
    weather_defaults = {f['api_name']: f['default'] for f in metadata_values}
    weather = fix_indoor_weather(games, weather, weather_defaults)
    for feature in metadata_values:
        col = feature['api_name']
        feature_data = weather[['game_id', col]]
        feature_data.to_csv(f"{output_dir}/{col}.csv", index=False)
    return
