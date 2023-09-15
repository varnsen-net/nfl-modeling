import numpy as np
import pandas as pd


def make_precision_map(features):
    """Make a precision map from a config file.
    
    :param dict features: A dictionary of features from a config file
    :return: A mapping of column names to precision
    :rtype: dict
    """
    precision_map = {}
    for category in features:
        for feature in features[category]:
            print(category, feature)
            precision_map[feature] = features[category][feature]['precision']
    return precision_map


def set_precision(df, precision_map):
    """Set the precision for certain columns in a training or test set.

    :param pd.DataFrame df: A set of training or test data
    :param dict precision_map: A dictionary mapping column names to precision
    """
    for col, precision in precision_map.items():
        feature = df.filter(regex=f"{col}$")
        df[feature.columns] = feature.round(precision)
    return df



if __name__ == '__main__':
    import json
    with open('./src/config.json', 'r') as f:
        config = json.load(f)
        features = config['features']
    print(features)
    precision_map = make_precision_map(features)
    train = pd.read_csv('./data/train/train.csv')
    processed_train = set_precision(train, precision_map)
    print(processed_train)
