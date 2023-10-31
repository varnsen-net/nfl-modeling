"""Unit tests for utils.py."""

import pytest

import numpy as np
import pandas as pd

from src.utils import get_kickoff_hours, shift_week_number


class TestGetKickoffHours:
    """Tests for get_kickoff_hours."""

    @pytest.fixture
    def gametimes(self):
        """A fixture for the gametimes column.
        
        :return: a column of game times
        """
        return pd.Series(['00:00', '02:00', '12:00', '23:59'])

    def test_standard_case(self, gametimes):
        """Test the standard case.
    
        :param pd.Series gametimes: a column of game times
        """
        hours = get_kickoff_hours(gametimes)
        assert hours.equals(pd.Series([0, 2, 12, 23]))

    def test_none_case(self):
        """Test the case where gametimes contains a None."""
        with pytest.raises(TypeError):
            get_kickoff_hours(pd.Series([None]))

    def test_nan_case(self):
        """Test the case where gametimes contains a NaN."""
        with pytest.raises(AttributeError):
            get_kickoff_hours(pd.Series([np.nan]))


class TestShiftWeekNumber:
    """Tests for shift_week_number."""

    @pytest.fixture
    def season_team_idx(self):
        """A fixture for the season_team_idx column.
        
        :return: 
        """
        df = pd.DataFrame({'season': [2023]*6,
                           'team': ['KC', 'KC', 'KC', 'SF', 'SF', 'SF'],
                           'week': [1, 2, 3, 1, 2, 3],
                           'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        df = df.set_index(['season', 'team', 'week'])
        return df

    def test_standard_case(self, season_team_idx):
        """Test the standard case.
    
        :param pd.DataFrame season_team_idx: a column of game times
        """
        shifted = shift_week_number(season_team_idx, 1)
        expected = pd.DataFrame({'season': [2023]*4,
                                 'team': ['KC', 'KC', 'SF', 'SF'],
                                 'week': [2, 3, 2, 3],
                                 'value': [1.0, 2.0, 4.0, 5.0]})
        expected = expected.set_index(['season', 'team', 'week'])
        assert shifted.equals(expected)
