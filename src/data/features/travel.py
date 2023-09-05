import pandas as pd
import numpy as np

from src.utils import parse_common_args


def get_city_coordinates(teams, loc_replacements):
    """Returns a dataframe with the decimal latitude and longitude for every
    NFL team city in the Sharpe teams data.

    Mostly saving this just for reference. I only need to run it once.
    
    :param pd.DataFrame teams: sharpe teams dataframe
    :param dict loc_replacements: dictionary of team location replacements
    :return: dataframe with city, lat, and lon columns
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

    :param np.array home_coords: home lat/lon coordinates
    :param np.array away_coords: away lat/lon coordinates
    :return: distances for each pair of lat/lon coordinates (km)
    :rtype: np.array
    """
    R = 6373.0
    lat1 = np.radians(home_coords[:, 0])
    lon1 = np.radians(home_coords[:, 1])
    lat2 = np.radians(away_coords[:, 0])
    lon2 = np.radians(away_coords[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * \
        np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def attach_lats_lons(games, city_coords):
    """Merge the city coordinates onto the games dataframe.
    
    :param pd.DataFrame games: raw games data
    :param pd.DataFrame city_coords: city lat/lon coordinates
    :return: games dataframe with home and away coordinates attached
    :rtype: pd.DataFrame
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


def get_away_travel_distances(games):
    """Calculate the distance traveled by the away team for each game.
    
    :param pd.DataFrame games: raw games data
    :return: away team travel features
    :rtype: pd.DataFrame
    """
    home_coords = games[['lat_home', 'lon_home']].values
    away_coords = games[['lat_away', 'lon_away']].values
    games['away_travel_distance_km'] = calculate_distances(home_coords, away_coords)
    away_travel_distances = games[['game_id', 'away_travel_distance_km']]
    return away_travel_distances


def make_travel_features(games, city_coords, travel_feat_path):
    """Build engineered features for team travel.

    :param pd.DataFrame games: raw games data
    :param pd.DataFrame city_coords: city coordinates
    :param str travel_feat_path: path to save travel features
    :return: None
    :rtype: None
    """
    games = attach_lats_lons(games, city_coords)
    away_travel_distances = get_away_travel_distances(games)
    away_travel_distances.to_csv(f"{travel_feat_path}/away-travel-distances.csv",
                                 index=False)
    return


if __name__ == '__main__':
    args = parse_common_args()
    raw_games_path = args.g
    city_coords_path = args.cc
    travel_feat_path = args.o

    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    city_coords = pd.read_csv(city_coords_path)
    make_travel_features(games, city_coords, travel_feat_path)
             

