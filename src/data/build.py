"""Build the full set of training and testing data from scratch.

Receives a set of paths from setup.sh, then fetches raw data, builds features, and assembles training data. Then it splits holdout data by year and writes to disk.
"""

import os
import json

import numpy as np
import pandas as pd

from src.utils import collect_setup_args
from src.data.raw.games import refresh_games_data
from src.data.raw.weather import refresh_weather_data
from src.data.train.train import build_train
from src.data.train.target import build_target


def split_data(df, holdout_year):
    """Split data into training and holdout sets.

    :param pd.DataFrame df: training or target data
    :param int holdout_year: starting season of holdout data (inclusive)
    :return: training and holdout data
    :rtype: tuple[pd.DataFrame]
    """
    train = df.loc[df["season"] < holdout_year]
    holdout = df.loc[df["season"] >= holdout_year]
    return train, holdout


if __name__ == '__main__':
    args = collect_setup_args()
    config_path = args.c
    raw_games_path = args.g
    raw_weather_path = args.w
    city_coords_path = args.cc
    features_path = args.f
    train_path = args.tr
    test_path = args.te


    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
        features_metadata = config['features']
        weather_metadata = features_metadata['weather']
        games_cols = config['training']['games_cols']
        holdout_year = config['training']['holdout_year']


    # Refresh raw data
    print('Refreshing raw games data...')
    refresh_games_data(raw_games_path)
    print('Refreshing raw weather data...')
    refresh_weather_data(weather_metadata, raw_games_path, raw_weather_path,
                         city_coords_path, batch_size=1000)


    # Build features
    for file in os.listdir('src/data/features'):
        if file.endswith('.py'):
            print(f'Building features with {file}...')
            module = file[:-3]
            exec(f'from src.data.features.{module} import build_features')
            output_dir = f"{features_path}/{module}"
            os.makedirs(output_dir, exist_ok=True)
            metadata = features_metadata[module]
            build_features(metadata=metadata,
                           raw_games_path=raw_games_path,
                           raw_weather_path=raw_weather_path,
                           city_coords_path=city_coords_path,
                           output_dir=output_dir)


    # Build training and test data
    print('Building training and test data...')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    full_train = build_train(games_cols, raw_games_path, features_path)
    train, train_holdout = split_data(full_train, holdout_year)
    full_target = build_target(raw_games_path, full_train)
    target = (full_target
              .loc[full_target["game_id"].isin(train["game_id"])]
              .sort_values('game_id'))
    target_holdout = (full_target
                      .loc[full_target["game_id"].isin(train_holdout["game_id"])]
                      .sort_values('game_id'))
    train.sort_values('game_id').to_csv(f'{train_path}/train.csv', index=False)
    target.to_csv(f'{train_path}/target.csv', index=False)
    train_holdout.sort_values('game_id').to_csv(f'{test_path}/test.csv', index=False)
    target_holdout.to_csv(f'{test_path}/target.csv', index=False)

