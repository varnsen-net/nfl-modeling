"""Helper functions for fetching raw NFL play-by-play data.

https://github.com/nflverse/nflverse-data/releases/tag/pbp
"""

import os

import numpy as np
import pandas as pd
import requests


def fetch_pbp_data(path, year):
    """Fetch raw play-by-play data for a given year and write to disk.
    
    :param str path: dir to write data to
    :param int year: season to fetch data for
    :return: None
    :rtype: None
    """
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
    r = requests.get(url)
    with open(f"{path}/play_by_play_{year}.parquet", "wb") as f:
        f.write(r.content)
    return


def fetch_missing_data(path, current_season):
    """Fetch missing play-by-play data for years prior to current season.
    
    :param str path: dir to write data to
    :param int current_season: current nfl season
    :return: None
    :rtype: None
    """
    years = range(1999, current_season)
    files = os.listdir(path)
    for year in years:
        if f"play_by_play_{year}.parquet" not in files:
            fetch_pbp_data(path, year)
    return


def refresh_plays_data(path, current_season):
    """Refresh raw play-by-play data if necessary.
    
    :param str path: dir to write data to
    :param int current_season: current nfl season
    :return: None
    :rtype: None
    """
    os.makedirs(path, exist_ok=True)
    fetch_missing_data(path, current_season)
    fetch_pbp_data(path, current_season)
    return
