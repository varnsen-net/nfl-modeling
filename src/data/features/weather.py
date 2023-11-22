"""Helper functions for building weather features."""

import csv

import numpy as np
import pandas as pd
import requests


def fix_indoor_weather(games, weather, defaults):
    """Fix the weather data for indoor games.
    
    :param games: *pd.DataFrame of shape (n_samples, n_features)*
        Raw games data.
    :param weather: *pd.DataFrame of shape (n_samples, n_features)*
        Raw weather data.
    :param defaults: *dict*
        Default values for weather features.
    :return: *pd.DataFrame*
        Weather data with indoor games fixed.
    """
    indoor_labels = ['dome', 'closed']
    indoor_games = games.query('roof in @indoor_labels').set_index('game_id')
    indoor_weather = pd.DataFrame(defaults, index=indoor_games.index)
    weather = weather.set_index('game_id')
    weather.update(indoor_weather)
    weather = weather.reset_index()
    return weather


def build_weather_features(weather_metadata, raw_games_path, raw_weather_path,
                           output_dir):
    """Make weather features.
    
    :param weather_metadata: *dict*
        Metadata for weather features.
    :param raw_games_path: *str*
        Path to raw games data.
    :param raw_weather_path: *str*
        Path to raw weather data.
    :param output_dir: *str*
        Path to output directory.
    :return: *None*
    """
    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    weather = pd.read_csv(raw_weather_path)
    weather_features = weather_metadata.values()
    weather_defaults = {f['api_name']: f['default'] for f in weather_features}
    weather = fix_indoor_weather(games, weather, weather_defaults)
    for feature in weather_features:
        col = feature['api_name']
        feature_data = weather[['game_id', col]]
        feature_data.to_csv(f"{output_dir}/{col}.csv", index=False)
    return
