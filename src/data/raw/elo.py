"""Helper functions for fetching raw elo ratings from 538."""

import pandas as pd
import requests


def refresh_elo_data(path):
    """Fetches NFL ELO data from 538 and saves to disk.

    :param str path: Path to save data to
    :return: None
    :rtype: None
    """
    url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'w') as f:
            f.write(response.text)
    return
