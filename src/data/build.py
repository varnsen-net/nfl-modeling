import os
import json
import argparse

import numpy as np
import pandas as pd

from src.data.raw.games import refresh_games_data
from src.data.raw.weather import refresh_weather_data
from src.data.train.train import build_train
from src.data.train.target import build_target


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', help='path to config file')
    argparser.add_argument('-g', help='path to raw games data')
    argparser.add_argument('-w', help='path to raw weather data')
    argparser.add_argument('-cc', help='path to city coordinates')
    argparser.add_argument('-f', help='path to features data')
    argparser.add_argument('-tr', help='path to training data')
    argparser.add_argument('-ta', help='path to target data')
    args = argparser.parse_args()

    config_path = args.c
    raw_games_path = args.g
    raw_weather_path = args.w
    city_coords_path = args.cc
    features_path = args.f
    train_path = args.tr
    target_path = args.ta


    # Refresh raw data
    print('Refreshing raw games data...')
    refresh_games_data(raw_games_path)
    print('Refreshing raw weather data...')
    refresh_weather_data(config_path, raw_games_path, raw_weather_path,
                         city_coords_path, batch_size=1000)


    # Build features
    for file in os.listdir('src/data/features'):
        if file.endswith('.py'):
            print(f'Building features from {file}...')
            module = file[:-3]
            exec(f'from src.data.features.{module} import build_features')
            output_dir = f"{features_path}/{module}"
            os.makedirs(output_dir, exist_ok=True)
            build_features(config_path=config_path,
                           raw_games_path=raw_games_path,
                           raw_weather_path=raw_weather_path,
                           city_coords_path=city_coords_path,
                           output_dir=output_dir)


    # Build training data
    print('Building training data...')
    train_dir = train_path.rsplit('/', 1)[0]
    os.makedirs(train_dir, exist_ok=True)
    build_train(config_path, raw_games_path, features_path, train_path)
    build_target(raw_games_path, target_path)

