"""Helper functions for fetching and saving NFL game data."""

import requests


def refresh_games_data(path):
    """Fetches NFL game data going back to 1999 and saves to disk.

    https://github.com/nflverse/nfldata/blob/master/data/games.csv

    :param str path: Path to save data to
    :return: None
    :rtype: None
    """
    url = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'w') as f:
            f.write(response.text)
    return
