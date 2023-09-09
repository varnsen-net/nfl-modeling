import os
import json
import argparse

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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', help='path to games data')
    argparser.add_argument('-o', help='path to output data')
    args = argparser.parse_args()
    raw_games_path = args.g
    output_path = args.o

    games = pd.read_csv(raw_games_path)
    games['target'] = make_target_col(games)
    train = games[['game_id', 'target']]
    train.to_csv(output_path, index=False)
