"""Helper functions for fetching raw NFL play-by-play data.

https://github.com/nflverse/nflverse-data/releases/tag/pbp
"""

import os

import numpy as np
import pandas as pd
import requests

from src.utils import refresh_raw_data


def fetch_pbp_data(url, path, filename):
    """Fetch raw play-by-play data for a given season and write to disk.
    
    :param str url: url to fetch data from
    :param str path: dir to write data to
    :param str filename: name of file
    :return: None
    :rtype: None
    """
    url = f"{url}/{filename}"
    path = f"{path}/{filename}"
    refresh_raw_data(url, path)
    return


def find_missing_seasons(current_season, path):
    """Find any missing local pbp season files.
    
    :param int current_season: current nfl season
    :param str path: dir to search for files
    :yield: missing season filenames
    :rtype: str
    """
    years = range(1999, current_season+1)
    files = os.listdir(path)
    for season in years:
        filename = f"play_by_play_{season}.parquet"
        if filename not in files:
            yield filename
        elif season == current_season:
            yield filename


def refresh_plays_data(current_season, url, path):
    """Refresh raw play-by-play data if necessary.
    
    :param int current_season: current nfl season
    :param str url: url to fetch data from
    :param str path: dir to write data to
    :return: None
    :rtype: None
    """
    os.makedirs(path, exist_ok=True)
    for filename in find_missing_seasons(current_season, path):
        fetch_pbp_data(url, path, filename)
    return
