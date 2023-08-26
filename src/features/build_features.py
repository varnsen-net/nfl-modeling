import pandas as pd
import numpy as np

from src.config import (RAW_GAMES_PATH,
                        RAW_WEATHER_PATH,
                        CITY_COORDS_PATH)
from src.features.travel import make_travel_features
from src.features.stats import make_stats_features


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

    make_travel_features(games, city_coords)
    make_stats_features(games)
