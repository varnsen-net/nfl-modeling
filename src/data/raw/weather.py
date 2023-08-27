import json
import csv

import numpy as np
import pandas as pd
import requests

from src.utils import fix_game_times, get_kickoff_hours


def fetch_gameday_weather(latitude, longtiude, date):
    """Use the open meteo historical weather api to get weather data.
    
    :param float latitude: latitude of game location
    :param float longtiude: longitude of game location
    :param str date: date of game in YYYY-MM-DD format
    :return: hourly weather data for the date and location
    :rtype: list
    """
    payload = {'latitude': latitude,
               'longitude': longtiude,
               'start_date': date,
               'end_date': date,
               'hourly': 'temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,surface_pressure,rain,snowfall,cloudcover,windspeed_10m',
               'min': date,
               'max': date,
               'timezone': 'America/New_York'}
    response = requests.get('https://archive-api.open-meteo.com/v1/archive',
                            params=payload)
    return response


def aggregate_weather_response(response, kickoff_hour):
    """Aggregates the weather response from open meteo over a four hour period.

    :param requests.models.Response response: response from open meteo api
    :param int kickoff_hour: hour when kickoff takes place
    :return: aggregated weather data for the four hour period after kickoff
    """
    data = response.json()['hourly']
    data = pd.DataFrame(data).iloc[kickoff_hour:kickoff_hour+4,:]
    agg = data.aggregate({'temperature_2m': 'mean',
                          'apparent_temperature': 'mean',
                          'relativehumidity_2m': 'mean',
                          'dewpoint_2m': 'mean',
                          'surface_pressure': 'mean',
                          'windspeed_10m': 'mean',
                          'cloudcover': 'mean',
                          'rain': 'sum',
                          'snowfall': 'sum'})
    agg = agg.round(2)
    return agg


def fetch_missing_weather(weather, raw_weather_path, batch_size=500):
    """Fetches weather data for each game in weather and appends to weather.csv.

    :param pd.DataFrame weather: weather dataframe
    :param str raw_weather_path: path to raw data directory
    :param int batch_size: number of games to fetch weather for at a time
    :return: None
    :rtype: None
    """
    weather_cols = ['temp_C', 'feelslike_C', 'rel_humidity_pc', 'dewpoint_C',
                    'surface_pressure_hPa', 'wind_speed_kmh', 'cloudcover_pc',
                    'rain_mm', 'snowfall_cm']
    weather = weather.set_index('game_id')
    print(f"{weather.shape[0]} remaining games with missing weather data. Current batch size is set to {batch_size}.")
    for index, row in weather.iloc[:batch_size,:].iterrows():
        date = row['gameday']
        kickoff_hour = row['kickoff_hour']
        latitude = row['lat']
        longitude = row['lon']
        print(f"Fetching weather for {latitude} by {longitude} on {date}...")
        response = fetch_gameday_weather(latitude, longitude, date)
        agg = aggregate_weather_response(response, kickoff_hour).to_list()
        with open(raw_weather_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([index] + agg)
    return


def refresh_weather_data(games, weather, city_coords, raw_weather_path):
    """Fetches weather data for each NFL game not found in weather.csv, then
    appends to file.

    :param pd.DataFrame games: raw games dataframe
    :param pd.DataFrame weather: raw weather dataframe
    :param pd.DataFrame city_coords: coordinates for each team's home city
    :param str raw_weather_path: path to raw data directory
    :return: None
    :rtype: None
    """
    missing_weather = (games
                       .merge(weather, on='game_id', how='left')
                       .query('temp_C.isna()')
                       .merge(city_coords[['name', 'lat', 'lon']], 
                              left_on='home_team', right_on='name', how='left'))
    if missing_weather.shape[0] > 0:
        print("Updating weather file...")
        start_times = fix_game_times(missing_weather)
        missing_weather['kickoff_hour'] = get_kickoff_hours(start_times)
        fetch_missing_weather(missing_weather, raw_weather_path)
    return



