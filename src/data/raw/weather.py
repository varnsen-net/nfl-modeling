import json
import csv
import argparse

import numpy as np
import pandas as pd
import requests

from src.utils import fix_game_times, get_kickoff_hours, parse_common_args


def fetch_gameday_weather(latitude, longtiude, date, weather_vars):
    """Use the open meteo historical weather api to get weather data for a
    single game.

    see https://open-meteo.com/en/docs/historical-weather-api
    
    :param float latitude: latitude of game location
    :param float longtiude: longitude of game location
    :param str date: date of game in YYYY-MM-DD format
    :param str weather_vars: comma-separated weather variables to fetch
        e.g. 'var1,var2,var3'
    :return: hourly weather data for the date and location
    :rtype: requests response object
    """
    payload = {'latitude': latitude,
               'longitude': longtiude,
               'start_date': date,
               'end_date': date,
               'hourly': weather_vars,
               'min': date,
               'max': date,
               'timezone': 'America/New_York'}
    response = requests.get('https://archive-api.open-meteo.com/v1/archive',
                            params=payload)
    return response


def aggregate_weather_response(response, kickoff_hour, aggregations):
    """Aggregates the weather response from open meteo over a four hour period.

    :param requests.models.Response response: response from open meteo api
    :param int kickoff_hour: hour when kickoff takes place
    :return: aggregated weather data for the four hour period after kickoff
    """
    data = response.json()['hourly']
    data = pd.DataFrame(data).iloc[kickoff_hour:kickoff_hour+4,:]
    agg = data.aggregate(aggregations)
    return agg


def find_games_missing_weather(games, weather):
    """Finds games in the raw games data that do not appear in the weather
    data.
    
    :param pd.DataFrame games: raw games dataframe
    :param pd.DataFrame weather: raw weather dataframe
    :return: rows from games that do not appear in weather
    :rtype: pd.DataFrame
    """
    games_missing_weather = games[~games['game_id'].isin(weather['game_id'])]
    return games_missing_weather


def attach_city_coords(games, city_coords):
    """Attaches the coordinates of each team's home city to the games df.
    
    :param pd.DataFrame games: raw games dataframe
    :param pd.DataFrame city_coords: coordinates for each team's home city
    :return: games dataframe with coordinates attached
    :rtype: pd.DataFrame
    """
    games = games.merge(city_coords[['name', 'lat', 'lon']], 
                        left_on='home_team', right_on='name', how='left')
    games = games.drop(columns=['name'])
    return games


def fetch_missing_weather(weather, raw_weather_path, weather_vars,
                          aggregations, batch_size):
    """Fetches weather data for each game in weather and appends to weather.csv.

    :param pd.DataFrame weather: weather dataframe
    :param str raw_weather_path: path to raw data directory
    :param str weather_vars: comma-separated weather variables to fetch
    :param int batch_size: number of games to fetch weather for at a time
    :return: None
    :rtype: None
    """
    print(f"{weather.shape[0]} remaining games with missing weather data. Current batch size is set to {batch_size}.")
    for row in weather.iloc[:batch_size,:].itertuples():
        game_id = row.game_id
        date = row.gameday
        kickoff_hour = row.kickoff_hour
        latitude = row.lat
        longitude = row.lon
        print(f"Fetching weather for {latitude} by {longitude} on {date}...")
        response = fetch_gameday_weather(latitude, longitude, date, weather_vars)
        agg = aggregate_weather_response(response, kickoff_hour, aggregations).to_list()
        with open(raw_weather_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([game_id] + agg)
    return


def refresh_weather_data(games, weather, city_coords, raw_weather_path,
                         weather_metadata, batch_size):
    """Fetches any missing weather data.

    Extract aggregations and weather variables from weather_metadata, then
    check for any games with missing weather data. If there are any, fetch
    weather data for those games.

    :param pd.DataFrame games: raw games dataframe
    :param pd.DataFrame weather: raw weather dataframe
    :param pd.DataFrame city_coords: coordinates for each team's home city
    :param str raw_weather_path: path to raw data directory
    :param dict weather_metadata: metadata for weather features
    :param int batch_size: number of games to fetch weather for at a time
    :return: None
    :rtype: None
    """
    aggregations = {f['api_name']: f['agg_func'] for f in weather_metadata.values()}
    weather_vars = ','.join(aggregations.keys())
    missing_weather = find_games_missing_weather(games, weather)
    missing_weather = attach_city_coords(missing_weather, city_coords)
    if missing_weather.shape[0] > 0:
        print("Updating weather file...")
        start_times = fix_game_times(missing_weather)
        missing_weather['kickoff_hour'] = get_kickoff_hours(start_times)
        fetch_missing_weather(missing_weather, raw_weather_path, weather_vars,
                              aggregations, batch_size)
    return


if __name__ == '__main__':
    args = parse_common_args()
    config_path = args.c
    raw_games_path = args.g
    raw_weather_path = args.w
    city_coords_path = args.cc

    with open(config_path) as f:
        config = json.load(f)
        weather_metadata = config['features']['weather']

    games = (pd.read_csv(raw_games_path)
             .dropna(subset=['result']))
    weather = pd.read_csv(raw_weather_path)
    city_coords = pd.read_csv(city_coords_path)
    refresh_weather_data(games, weather, city_coords, raw_weather_path,
                         weather_metadata, batch_size=1000)
             
