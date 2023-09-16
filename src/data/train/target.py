"""Helper functions for building target column for games data."""

import numpy as np
import pandas as pd


def make_target_col(games):
    """Make target column for games data.
    
    :param pd.DataFrame games: games data
    :return: games data with target column
    :rtype: pd.DataFrame
    """
    target = np.where(games['result'] > 0, 1, 0)
    return target


def build_target(raw_games_path, target_path):
    """Build target column for games data.
    
    :param str raw_games_path: path to raw games data
    :param str target_path: path to output data
    :return: None
    :rtype: None
    """
    games = pd.read_csv(raw_games_path)
    games['target'] = make_target_col(games)
    target = games[['game_id', 'target']]
    target.to_csv(target_path, index=False)
    return
