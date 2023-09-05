import numpy as np
import pandas as pd


def set_precision(df):
    """Set the precision for certain columns in a training or test set.

    Negative precision values indicate that the column should be rounded to
    the nearest 10^precision.
    
    :param pd.DataFrame df: A set of training or test data
    """
    precision_map = {
        'pythag_exp_away': 2,
        'pythag_exp_home': 2,
        'away_travel_distance_km': -1,
        'temp_C': 0,
        'feelslike_C': 0}
    for col, precision in precision_map.items():
        df[col] = df[col].round(precision)
    return df



if __name__ == '__main__':
    train = pd.read_csv('./data/training/train.csv')
    processed = set_precision(train)
    print(processed[['pythag_exp_away', 'pythag_exp_home', 'away_travel_distance_km', 'temp_C', 'feelslike_C']])


