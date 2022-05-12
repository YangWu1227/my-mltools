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
                                   InvalidIdentifierError,
                                   InvalidColumnDtypeError)

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

# ---------------------------------------------------------------------------- #
#                     Test data for the rest of the module                     #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope='module')
def test_data():
    return pd.DataFrame({
        'categorical1': ('A', 'A', 'B', 'C', 'D', 'D', 'C', pd.NA, 'C', 'E'),
        'categorical2': ('bachelor', 'highschool', pd.NA, 'grad', 'grad', pd.NA, 'highschool', 'college', 'college', 'bachelor'),
        'case_convert': ('Upper',) * 5 + ('lower',) * 3 + (pd.NA, pd.NA),
        'invalid_case_convert': tuple(range(0, 10))
    })

# ---------------------------------------------------------------------------- #
#                        Tests for case_convert function                       #
# ---------------------------------------------------------------------------- #


class TestCaseConvert:
    """
    Tests for the case_convert helper function.
    """

    # --------------------- Tests that exceptions are raised --------------------- #

    def test_case_convert_errors(self, test_data):
        """
        Tests that case_convert raises exceptions when 'cols' and 'to' are passed invalid inputs.
        """

        # Range offset for 'cols'
        with pytest.raises(TypeError, match="'cols' must be a sequence like a list or a single string"):
            cu.case_convert(df=test_data, cols=range(0, 3))

        # Invalid inputs for 'to'
        # Numeric
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            cu.case_convert(df=test_data, cols='case_convert', to=3)
        # Boolean
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            cu.case_convert(df=test_data, cols='case_convert', to=True)
        # List
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            cu.case_convert(df=test_data, cols='case_convert',
                            to=['upper'])
        # Single element tuple
        with pytest.raises(ValueError, match="'to' must either by 'lower', 'upper', 'title', or 'capitalize'"):
            cu.case_convert(
                df=test_data, cols='case_convert', to=('lower', ))

    # ------------------- Tests that custom exception is raised ------------------ #

    def test_case_convert_custom_error(self, test_data):
        """
        Test that when user passes non 'string' columns in 'cols' the function raises InvalidColumnDtypeError.
        """

        with pytest.raises(InvalidColumnDtypeError, match=escape("Columns ['invalid_case_convert'] are invalid as dtype 'string' is expected")):
            cu.case_convert(
                test_data, ['invalid_case_convert', 'case_convert'])

    # -------------------------- Tests for functionality ------------------------- #

    @pytest.mark.parametrize(
        "cols, to",
        [
            # No user supplied columns
            (None, 'lower'),
            (None, 'upper'),
            (None, 'title'),
            (None, 'capitalize'),
            # User supplied columns
            ('case_convert', 'lower'),
            (['case_convert', 'categorical1', 'categorical2'], 'upper'),
            ('case_convert', 'title'),
            (['case_convert', 'categorical2'], 'capitalize')
        ],
        scope='function'
    )
    def test_case_convert(self, test_data, cols, to):
        """
        Test that case_convert returns expected results given inputs with branches.
        """

        # Test all branches return correct class
        type(cu.case_convert(test_data, cols=cols, to=to)) == type(pd.DataFrame())

    def test_case_convert_branch(self, test_data):
        """
        Test that three possible combinations of parameters of case_convert.
        """
        # Test two branches
        branch1 = cu.case_convert(test_data, cols=None, to='lower')
        branch2 = cu.case_convert(
            test_data, cols=['case_convert', 'categorical2'], to='upper')

        # Branch 1
        pd.testing.assert_frame_equal(
            left=branch1,
            right=pd.DataFrame({
                'categorical1': ('a', 'a', 'b', 'c', 'd', 'd', 'c', pd.NA, 'c', 'e'),
                'categorical2': ('bachelor', 'highschool', pd.NA, 'grad', 'grad', pd.NA, 'highschool', 'college', 'college', 'bachelor'),
                'case_convert': ('upper',) * 5 + ('lower',) * 3 + (pd.NA, pd.NA),
                'invalid_case_convert': tuple(range(0, 10))
            })
        )

        # Branch 2
        pd.testing.assert_frame_equal(
            left=branch2,
            right=pd.DataFrame({
                'categorical1': ('A', 'A', 'B', 'C', 'D', 'D', 'C', pd.NA, 'C', 'E'),
                'categorical2': ('BACHELOR', 'HIGHSCHOOL', pd.NA, 'GRAD', 'GRAD', pd.NA, 'HIGHSCHOOL', 'COLLEGE', 'COLLEGE', 'BACHELOR'),
                'case_convert': ('UPPER',) * 5 + ('LOWER',) * 3 + (pd.NA, pd.NA),
                'invalid_case_convert': tuple(range(0, 10))
            })
        )

        # Branch 3 (modify test data in place)
        copied_test_data = test_data.copy()
        # No assignment
        cu.case_convert(copied_test_data, cols='categorical2',
                        to='title', inplace=True)

        # Test data should be modified
        pd.testing.assert_frame_equal(
            left=copied_test_data,
            right=pd.DataFrame({
                'categorical1': ('A', 'A', 'B', 'C', 'D', 'D', 'C', pd.NA, 'C', 'E'),
                'categorical2': ('Bachelor', 'Highschool', pd.NA, 'Grad', 'Grad', pd.NA, 'Highschool', 'College', 'College', 'Bachelor'),
                'case_convert': ('Upper',) * 5 + ('lower',) * 3 + (pd.NA, pd.NA),
                'invalid_case_convert': tuple(range(0, 10))
            })
        )

# ---------------------------------------------------------------------------- #
#                          Tests for relocate function                         #
# ---------------------------------------------------------------------------- #


class TestRelocate:
    """
    Tests for the relocate function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_relocate_errors(self, test_data):
        """
        Test that relocate raises exceptions when arguments are passed invalid inputs.
        """

        # Exceptions for 'to_move'
        with pytest.raises(TypeError, match="'to_move' must be a sequence like a list or a single string"):
            cu.relocate(
                test_data,
                # Cannot use integer index
                to_move=2,
                after='case_convert'
            )
        # Keyword error if supplied a list of non-string objects
        with pytest.raises(KeyError, match="not in index"):
            cu.relocate(
                test_data,
                # A list of non-strings
                to_move=[3, 2],
                before='case_convert'
            )
        # Keyword error if supplied a invalid column names
        with pytest.raises(KeyError, match="not in index"):
            cu.relocate(
                test_data,
                # Invalid columns name
                to_move=['non-existent', 'case_convert'],
                after='case_convert'
            )
        # Exceptions for 'after' or 'before
        with pytest.raises(TypeError, match="must supply only one of 'before' and 'after' as a string"):
            cu.relocate(
                test_data,
                to_move=['case_convert'],
                # Both specified
                before='categorical1',
                after='categorical2'
            )
        with pytest.raises(TypeError, match="must supply only one of 'before' and 'after' as a string"):
            cu.relocate(
                test_data,
                # Both 'after' and 'before are None
                to_move='case_convert'
            )
        # Wrong input for either 'before' or 'after'
        with pytest.raises(TypeError, match="must supply only one of 'before' and 'after' as a string"):
            cu.relocate(
                test_data,
                to_move=['case_convert'],
                after=True
            )
        with pytest.raises(TypeError, match="must supply only one of 'before' and 'after' as a string"):
            cu.relocate(
                test_data,
                to_move=['case_convert'],
                before=[False, True]
            )

    # -------------------------- Tests for functionality ------------------------- #

    @pytest.mark.parametrize(
        "to_move, before, after, result",
        [
            # List of columns move before
            (['categorical2', 'case_convert'], 'categorical1', None, ['categorical2',
                                                                      'case_convert',
                                                                      'categorical1',
                                                                      'invalid_case_convert']),
            # Single column move after
            ('invalid_case_convert', None, 'categorical1', ['categorical1',
                                                            'invalid_case_convert',
                                                            'categorical2',
                                                            'case_convert']),
            # Columns to move contain reference columns should simply move the other columns
            (['categorical1', 'categorical2', 'case_convert'], None, 'case_convert', ['case_convert',
                                                                                      'categorical1',
                                                                                      'categorical2',
                                                                                      'invalid_case_convert']),
            (['categorical2', 'case_convert', 'invalid_case_convert'], 'categorical2', None, ['categorical1',
                                                                                              'case_convert',
                                                                                              'invalid_case_convert',
                                                                                              'categorical2']),
            # Edge cases where one of 'before' and 'after' is correctly specified as string but the other an invalid input
            # The expected behavior is that whichever one is correctly specified should be the reference column and the other ignored
            ('categorical1', True, 'case_convert', ['categorical2',
                                                    'case_convert',
                                                    'categorical1',
                                                    'invalid_case_convert']),
            ('invalid_case_convert', 'case_convert', 3, ['categorical1',
                                                         'categorical2',
                                                         'invalid_case_convert',
                                                         'case_convert'])
        ],
        scope='function'
    )
    # Original order is ['categorical1', 'categorical2', 'case_convert', 'invalid_case_convert']
    def test_relocate(self, test_data, to_move, before, after, result):
        """
        Test that relocate returns expected results given inputs with branches.
        """

        # Tests
        assert cu.relocate(test_data, to_move, before,
                           after).columns.tolist() == result

# ---------------------------------------------------------------------------- #
#                        Tests for find_missing function                       #
# ---------------------------------------------------------------------------- #


class TestFindMissing:
    """
    Tests for the find_missing helper function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_find_missing_errors(self, test_data):
        """
        Tests that find_missing raises exceptions when 'df' and 'axis' are passed invalid inputs.
        """

        # Errors for 'axis'
        # Usually type errors with message--- 'The argument 'axis' must be an integer'
        # Value error if supplied anything other than 0 or 1
        with pytest.raises(TypeError):
            cu.find_missing(test_data, axis={'3': 9})
        with pytest.raises(ValueError):
            cu.find_missing(test_data, axis=10)

    # -------------------------- Tests for functionality ------------------------- #

    def test_find_missing(self, test_data):
        """
        Test that find_missing returns expected results given inputs with branches.
        """

        # Return for rows
        pd.testing.assert_series_equal(
            left=cu.find_missing(test_data, axis=1),
            # The follong rows have missing values
            right=pd.Series(data=(True,) * 5, index=(2, 5, 7, 8, 9))
        )

        # Return for columns
        pd.testing.assert_series_equal(
            left=cu.find_missing(test_data, axis=0),
            right=pd.Series(
                data=(True,) * 3, index=('categorical1', 'categorical2', 'case_convert'))
        )

# ---------------------------------------------------------------------------- #
#                          Tests for freq_tbl function                         #
# ---------------------------------------------------------------------------- #


class TestFreqTable:
    """
    Tests for the freq_tbl helper function.
    """

    # ------------------------ Tests for exceptions raised ----------------------- #

    def test_freq_tbl_errors(self):
        """
        Check that freq_tbl() raises exceptions when function fails (non **kwargs).
        """

        # When 'dropna' is passed an invalid object whose boolean value is ambiguous
        with pytest.raises(TypeError, match='boolean value of NA is ambiguous'):
            cu.freq_tbl(pd.DataFrame({'col': ("A", "B", "A")}), pd.NA)
        # Sometimes 'dropna' may also result in value errors
        with pytest.raises(ValueError):
            cu.freq_tbl(pd.DataFrame(
                {'col': ("A", "B", "A")}), pd.Series(('A', 'B')))
        # Invalid input for 'cardinality' will raise a type error
        with pytest.raises(TypeError, match="'cardinality' must be an integer"):
            cu.freq_tbl(pd.DataFrame(
                {'col': ("A", "B", "A")}), False, 'non-integer')
        with pytest.raises(TypeError, match="'cardinality' must be an integer"):
            cu.freq_tbl(pd.DataFrame(
                {'col': ("A", "B", "A")}), False, 3.22)

    @pytest.mark.parametrize(
        "df, dropna",
        [(pd.DataFrame({'col': ("A", "B", "A")}), False)],
        scope='function'
    )
    def test_freq_tbl_kwargs(self, df, dropna):
        """
        Test that exception is raised when invalid **kwargs are provided to freq_tbl().
        """
        with pytest.raises(ValueError, match="Only 'normalize', 'sort', and 'ascending' are supported as extra keyword arguments"):
            cu.freq_tbl(df, dropna, wrong_keyword=True)
        with pytest.raises(ValueError, match="Only 'normalize', 'sort', and 'ascending' are supported as extra keyword arguments"):
            cu.freq_tbl(df, dropna, subset=True)

    # -------------------------- Tests for functionality ------------------------- #

    @pytest.mark.parametrize(
        "sort, normalize",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False)
        ],
        scope='function'
    )
    def test_freq_tbl(self, test_data, sort, normalize):
        """
        Test that freq_tbl() returns the correct class and length given a test dataframe and all combinations of parameters for 'sort' and 'normalize'.
        """

        # Outputs
        tbls = cu.freq_tbl(test_data, dropna=True,
                           sort=sort, normalize=normalize)

        # ------------- The overall test includes 'invalid_case_convert' ------------- #

        # Tuple
        assert isinstance(tbls, tuple) == True
        # Check '_fileds' attributes match test data string columns names
        assert tbls._fields == (
            'categorical1', 'categorical2', 'case_convert', 'invalid_case_convert')
        # Check length
        assert len(tbls) == 4

    def test_freq_tbl_branch(self, test_data):
        """
        Branch tests for freq_tbl and combinations of 'sort' and 'normalize' parameters.
        """
        # Exclude 'invalid_case_convert' (0 - 10 integers) from test data by setting cardinality to 6
        tbls_true_false = cu.freq_tbl(
            test_data, dropna=True, cardinality=6, sort=True, normalize=False)
        tbls_false_true = cu.freq_tbl(
            test_data, dropna=True, cardinality=6, sort=False, normalize=True)

        # Expected results for two of the four branches
        expected_index = {
            # Sort=True
            'sort_true': (
                ['C', 'A', 'D', 'B', 'E'],
                ['bachelor', 'highschool', 'grad', 'college'],
                ['Upper', 'lower']
            ),
            # Sort=False
            'sort_false': (
                ['A', 'B', 'C', 'D', 'E'],
                ['bachelor', 'highschool', 'grad', 'college'],
                ['Upper', 'lower'],
            )
        }
        expected_values = {
            # Normalize=False
            'normalize_false': [
                np.array([[3], [2], [2], [1], [1]]),
                np.array([[2], [2], [2], [2]]),
                np.array([[5], [3]])
            ],
            # Normalize=True
            'normalize_true': [
                np.array([[0.2], [0.1], [0.3], [0.2], [0.1]]),
                np.array([[0.2], [0.2], [0.2], [0.2]]),
                np.array([[0.6], [0.4]])
            ]
        }

        # ------------- The branch tests excludes 'invalid_case_convert' ------------- #

        # Branch (sort=True and normalize=False)
        for num, tbl in enumerate(tbls_true_false):
            assert all(tbl.index == expected_index['sort_true'][num])
            assert all(tbl.values ==
                       expected_values['normalize_false'][num])

        # Branch (sort=False and normalize=True)
        for num, tbl in enumerate(tbls_false_true):
            assert all(tbl.index == expected_index['sort_false'][num])
            assert all(tbl.round(1).values ==
                       expected_values['normalize_true'][num])
