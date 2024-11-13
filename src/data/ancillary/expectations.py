""""""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import PATHS


def preprocess_plays(raw_plays):
    """Preprocess raw play-by-play data.

    :param pd.DataFrame raw_plays: Raw play-by-play data.
    :return: Preprocessed play-by-play data.
    :rtype: pd.DataFrame
    """
    return (raw_plays
            .query('posteam != "" and posteam != "None"')
            .query('yrdln.notnull()'))


def fix_bad_yardlines(plays):
    """Fix bad yardlines in play-by-play data.
    
    :param pd.DataFrame plays: Play-by-play data.
    :return: New column with fixed yardlines.
    :rtype: pd.Series
    """
    return plays['yrdln'].where(plays['yrdln'] != '50', 'ZZZ 50')


def make_posteam_results(plays):
    """Make new results column where the sign of the result is relative to the
    posteam.
    
    :param pd.DataFrame plays: Play-by-play data.
    :return: New column with posteam results.
    :rtype: pd.Series
    """
    return plays['result'].where(plays['posteam_type'] == 'home',
                                 -plays['result'])


def calculate_net_value(plays):
    """Calculate the net value of all the plays.
    
    :param pd.DataFrame plays: Play-by-play data.
    :return: New column with net values.
    :rtype: pd.Series
    """
    return plays['posteam_result'] - (plays['posteam_score'] - plays['defteam_score'])


def make_absolute_yardlines(plays):
    """Convert relative yardlines to absolute yardlines.
    
    Yardlines in the plays data are given like 'KC 69'. This function converts
    them to absolute yardlines (i.e., 0-99).

    :param pd.DataFrame plays: Play-by-play data.
    :return: New column with absolute yardlines.
    :rtype: pd.Series
    """
    yard_lines = plays['yrdln_fixed'].str.split(' ', expand=True)
    yard_lines[1] = yard_lines[1].astype(int)
    cond = plays['posteam'] == yard_lines[0]
    absolute_yard_lines = yard_lines[1].where(cond, 100 - yard_lines[1])
    return absolute_yard_lines


def calulate_mean_yrdln_values(plays):
    """Calculate the mean net value of plays at each yardline.
    
    :param pd.DataFrame plays: Play-by-play data.
    :return: Mean net value of plays at each yardline.
    :rtype: pd.DataFrame
    """
    return (plays
            .groupby(['absolute_yrdln'])
            ['net_value']
            .mean())


if __name__ == '__main__':
    raw_plays_path = PATHS['raw_plays']
    expected_values_path = PATHS['expected_values']
    all_plays = pd.DataFrame()
    expected_values = pd.DataFrame()

    seasons = list(range(2001, 2025))
    all_plays = pd.concat([pd.read_parquet(raw_plays_path / f"play_by_play_{season}.parquet")
                           for season in seasons])
    all_plays = preprocess_plays(all_plays)

    types = ['home', 'away']
    downs = [1] # this is a list so we can add more downs later
    grid = [(type, down) for type in types for down in downs]

    for type, down in grid:
        plays = (all_plays
                 .query('posteam_type == @type')
                 .query('down == @down')
                 .assign(yrdln_fixed=fix_bad_yardlines,
                         posteam_result=make_posteam_results,
                         net_value=calculate_net_value,
                         absolute_yrdln=make_absolute_yardlines))

        plays = plays[['posteam', 'posteam_type', 'posteam_score', 'defteam_score',
                      'result', 'posteam_result', 'yrdln_fixed', 'net_value',
                      'absolute_yrdln']]

        yrdln_values = calulate_mean_yrdln_values(plays)
        yrdln_values = (yrdln_values
                        .reset_index()
                        .assign(posteam_type=type,
                                down=down))


        regr = LinearRegression()
        regr.fit(yrdln_values[['absolute_yrdln']], yrdln_values['net_value'])
        yrdln_values['expected_value'] = regr.predict(yrdln_values[['absolute_yrdln']])
        expected_values = pd.concat([expected_values, yrdln_values])

    print(expected_values)

    expected_values.to_csv(expected_values_path, index=False)

    fig, ax = plt.subplots()
    sns.scatterplot(data=expected_values, x='absolute_yrdln', y='net_value',
                    hue='posteam_type', ax=ax)
    sns.lineplot(data=expected_values, x='absolute_yrdln', y='expected_value',
                 hue='posteam_type', ax=ax)
    plt.tight_layout()
    plt.show()
