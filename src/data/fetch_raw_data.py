import requests

import pandas as pd

from src.data.weather import refresh_weather_data
from src.config import (RAW_GAMES_PATH,
                        RAW_WEATHER_PATH,
                        CITY_COORDS_PATH)

def refresh_games_data():
    """Fetches NFL game data going back to 1999 and saves to disk.

    https://github.com/nflverse/nfldata/blob/master/data/games.csv

    :return: None
    :rtype: None
    """
    url = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    response = requests.get(url)
    with open(RAW_GAMES_PATH, 'w') as f:
        f.write(response.text)
    return


if __name__ == '__main__':
    refresh_games_data()

    games = (pd.read_csv(RAW_GAMES_PATH)
             .dropna(subset=['result']))
    weather = pd.read_csv(RAW_WEATHER_PATH)
    city_coords = pd.read_csv(CITY_COORDS_PATH)
    refresh_weather_data(games, weather, city_coords)
             

