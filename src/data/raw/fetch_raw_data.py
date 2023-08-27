import json

import pandas as pd
import requests

from src.data.raw.weather import refresh_weather_data


with open('config.json') as f:
    config = json.load(f)
    RAW_GAMES_PATH = config['paths']['raw_games']
    RAW_WEATHER_PATH = config['paths']['raw_weather']
    CITY_COORDS_PATH = config['paths']['city_coords']


def refresh_games_data(path=RAW_GAMES_PATH):
    """Fetches NFL game data going back to 1999 and saves to disk.

    https://github.com/nflverse/nfldata/blob/master/data/games.csv

    :param str path: Path to save data to
    :return: None
    :rtype: None
    """
    url = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    response = requests.get(url)
    with open(f"{path}/games.csv", 'w') as f:
        f.write(response.text)
    return


if __name__ == '__main__':
    refresh_games_data()

    games = (pd.read_csv(f"{RAW_GAMES_PATH}/games.csv")
             .dropna(subset=['result']))
    weather = pd.read_csv(f"{RAW_WEATHER_PATH}/weather.csv")
    city_coords = pd.read_csv(f"{CITY_COORDS_PATH}/city-coordinates.csv")
    refresh_weather_data(games, weather, city_coords)
             

