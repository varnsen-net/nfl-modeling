"""Helper functions for building engineered features for team travel."""

import json

import pandas as pd
import numpy as np


def get_city_coordinates(teams, loc_replacements):
    """Returns a dataframe with the decimal latitude and longitude for every
    NFL team city in the Sharpe teams data.

    Mostly saving this just for reference. I only need to run it once.
    
    :param pd.DataFrame teams: df with team names and locations
    :param dict loc_replacements: dict of location replacements
    :return: city names, latitudes, and longitudes
    :rtype: pd.DataFrame
    """
    cities = teams['location'].unique()
    cities = pd.Series(cities, name='location')
    cities_adj = cities.replace(loc_replacements)
    lat = []
    lon = []
    for city in cities_adj:
        print(city)
        url = f'https://nominatim.openstreetmap.org/search?q={city}&format=json'
        response = requests.get(url).json()
        lat.append(response[0]['lat'])
        lon.append(response[0]['lon'])
    city_coords = pd.DataFrame({'city':cities, 'lat':lat, 'lon':lon})
    city_coords = pd.concat([cities, city_coords[['lat', 'lon']]], axis=1)
    return city_coords


def calculate_distances(home_coords, away_coords):
    """Calculates the distance between two arrays of lat/lon coordinate pairs.

    :param home_coords: Home lat/lon coordinates.
    :type home_coords: np.array of shape (n, 2)
    :param away_coords: Away lat/lon coordinates.
    :type away_coords: np.array of shape (n, 2)
    :return: Distance between the two coordinates in km.
    :rtype: np.array of shape (n, 1)
    """
    R = 6373.0
    lat1 = np.radians(home_coords[:, 0])
    lon1 = np.radians(home_coords[:, 1])
    lat2 = np.radians(away_coords[:, 0])
    lon2 = np.radians(away_coords[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def attach_lats_lons(games, city_coords):
    """Merge the city coordinates onto the games dataframe.
    
    :param games: Raw games data.
    :type games: pd.DataFrame of shape (n_rows, n_cols)
    :param city_coords: City coordinates data.
    :type city_coords: pd.DataFrame of shape (n_teams, n_cols)
    :return: Games dataframe with city coordinates attached.
    :rtype: pd.DataFrame of shape (n_rows, n_cols)
    """
    games = (games
             .merge(city_coords[['name', 'lat', 'lon']], 
                    left_on='away_team', right_on='name', how='left')
             .drop(columns=['name'])
             .merge(city_coords[['name', 'lat', 'lon']],
                    left_on='home_team', right_on='name', how='left',
                    suffixes=['_away', '_home'])
             .drop(columns=['name']))
    return games


def get_travel_distances(games):
    """Calculate the distance traveled by the away team for each game.
    
    :param games: Raw games data.
    :type games: pd.DataFrame of shape (n_rows, n_cols)
    :return: Dataframe with the travel distances for each game.
    :rtype: pd.DataFrame of shape (n_rows, n_cols)
    """
    travel_distances = games[['game_id']].set_index('game_id')
    away_distances = calculate_distances(games[['lat_home', 'lon_home']].values,
                                         games[['lat_away', 'lon_away']].values)
    travel_distances['away_travel_distance'] = away_distances
    travel_distances['home_travel_distance'] = 0
    return travel_distances


def get_coord_deltas(games):
    """Calculate the difference in longitude and latitude between the home and
    away teams.

    :param games: Raw games data.
    :type games: pd.DataFrame of shape (n_rows, n_cols)
    :return: Dataframe with the away team longitude deltas.
    :rtype: pd.DataFrame of shape (n_rows, n_cols)
    """
    travel_deltas = games[['game_id']].set_index('game_id')
    away_lon_delta = games['lon_home'] - games['lon_away']
    away_lat_delta = games['lat_home'] - games['lat_away']
    travel_deltas['away_lon_delta'] = away_lon_delta.values
    travel_deltas['away_lat_delta'] = away_lat_delta.values
    travel_deltas['home_lon_delta'] = 0
    travel_deltas['home_lat_delta'] = 0
    return travel_deltas


def build_travel_features(raw_games, city_coords):
    """Build engineered features for team travel.

    :param pd.DataFrame raw_games: Raw games data.
    :param pd.DataFrame city_coords: City coordinates data.
    :return: Dataframe with the engineered travel features.
    :rtype: pd.DataFrame of shape (n_rows, n_cols)
    """
    games = attach_lats_lons(raw_games, city_coords)
    travel_distances = get_travel_distances(games)
    lon_deltas = get_coord_deltas(games)
    travel_features = pd.concat([travel_distances, lon_deltas], axis=1)
    return travel_features
