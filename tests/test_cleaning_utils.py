# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import pytest

# ----------------------------- Standard library ----------------------------- #

import os
from re import escape

# ------------------------------- Intra-package ------------------------------ #

import my_mltools.cleaning_utils as cu
from my_mltools.exceptions import (ColumnNameKeyWordError,
                                   ColumnNameStartWithDigitError,
                                   InvalidIdentifierError)

# ---------------------------------------------------------------------------- #
#                        Tests for column names helpers                        #
# ---------------------------------------------------------------------------- #

# -------------------- Test data for column names helpers -------------------- #


@pytest.fixture(scope='class')
def check_col_nms_test_df():
    return (
        # Non-dataframe
        pd.Series((1, 2)),
        # Start with digit
        pd.DataFrame({'123col': (1, 2)}),
        # Keyword
        pd.DataFrame({'def': (1, 2)}),
        # Special characters
        pd.DataFrame({'^3edf': (1, 2)}),
        pd.DataFrame({'price ($)': (1, 2)}),
        pd.DataFrame({'percent%': (1, 2)}),
        # Trailing and leading White spaces
        pd.DataFrame({' trailing   leading    ': (1, 2)})
    )


class TestColumnNmsHelpers:
    """
    Check the exceptions raised when column names helper functions fail.
    """

    def test_check_col_nms(self, check_col_nms_test_df):
        """
        Check that check_col_nms() raises exceptions when data frames contain invalid identifiers.
        """
        # Non-dataframe
        with pytest.raises(TypeError, match="'df' must be a DataFrame"):
            cu.check_col_nms(check_col_nms_test_df[0])
        # Start with digit
        with pytest.raises(ColumnNameStartWithDigitError, match=escape("Columns ['123col'] must not start with digits")):
            cu.check_col_nms(check_col_nms_test_df[1])
        # Keyword
        with pytest.raises(ColumnNameKeyWordError, match=escape("Columns ['def'] are keywords of the language, and cannot be used as ordinary identifiers")):
            cu.check_col_nms(check_col_nms_test_df[2])
        # Special characters
        with pytest.raises(InvalidIdentifierError, match=escape("Columns ['^3edf'] are invalid identifiers")):
            cu.check_col_nms(check_col_nms_test_df[3])
        with pytest.raises(InvalidIdentifierError, match=escape("Columns ['price ($)'] are invalid identifiers")):
            cu.check_col_nms(check_col_nms_test_df[4])
        with pytest.raises(InvalidIdentifierError, match=escape("Columns ['percent%'] are invalid identifiers")):
            cu.check_col_nms(check_col_nms_test_df[5])
        with pytest.raises(InvalidIdentifierError, match=escape("Columns [' trailing   leading    '] are invalid identifiers")):
            cu.check_col_nms(check_col_nms_test_df[6])

    def test_clean_col_nms(self, check_col_nms_test_df):
        """
        Check that clean_col_nms() fixes invalid identifiers.
        """
        # Digits '123col' become 'col' since leading characters are moved until a letter is matched
        assert cu.clean_col_nms(
            check_col_nms_test_df[1]).columns.tolist() == ['col']
        # Digit and special character are removed from '^3edf'
        assert cu.clean_col_nms(
            check_col_nms_test_df[3]).columns.tolist() == ['edf']
        # White spaces and special characters are removed from 'price ($)' and 'percent%'
        assert cu.clean_col_nms(
            check_col_nms_test_df[4]).columns.tolist() == ['price']
        assert cu.clean_col_nms(
            check_col_nms_test_df[5]).columns.tolist() == ['percent']
        # Leading and trailing white spaces are removed first, then while spaces in between are replace with '_'
        assert cu.clean_col_nms(
            check_col_nms_test_df[6]).columns.tolist() == ['trailing_leading']
