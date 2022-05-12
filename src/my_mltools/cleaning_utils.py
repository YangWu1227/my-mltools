# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np

# ----------------------------- Standard library ----------------------------- #

import os
import sys
from itertools import compress
from re import sub
from keyword import iskeyword
from collections import namedtuple
from typing import List, Dict, Tuple, Union, Optional, NamedTuple

# ------------------------------- Intra-package ------------------------------ #

from my_mltools.exceptions import (ColumnNameKeyWordError,
                                   ColumnNameStartWithDigitError,
                                   InvalidIdentifierError,
                                   InvalidColumnDtypeError)
from my_mltools.check_utils import is_list_str, is_string_dtype_pd

# ---------------------------------------------------------------------------- #
#                               Cleaning helpers                               #
# ---------------------------------------------------------------------------- #

# ------------------ Function to check column name integrity ----------------- #


def check_col_nms(df: pd.DataFrame) -> None:
    """
    This function is a helper that checks the integrity of the column names. 

    Parameters
    ----------
    df : DataFrame

    Raises
    ------
    TypeError
        The argument 'df' must be a DataFrame.
    ColumnNameKeyWordError
        Columns name cannot contain reserved keywords like 'def', 'for'.
    ColumnNameStartWithDigitError
        Column names cannot begin with a digit.
    InvalidIdentifierError
        Catch-all exception for invalid identifiers.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a DataFrame")

    col_nms = df.columns.tolist()

    # Check column names are not keywords
    kw_index = [iskeyword(col) for col in col_nms]
    if any(kw_index):
        raise ColumnNameKeyWordError(list(compress(col_nms, kw_index)))

    # Check column names do not start with digit
    digit_index = [col[0].isdigit() for col in col_nms]
    if any(digit_index):
        raise ColumnNameStartWithDigitError(
            list(compress(col_nms, digit_index)))

    # Catch-all check
    invalid_identifier_index = [not col.isidentifier() for col in col_nms]
    if any(invalid_identifier_index):
        raise InvalidIdentifierError(
            list(compress(col_nms, invalid_identifier_index)))


# --------------------- Function that cleans column names -------------------- #


def clean_col_nms(df: pd.DataFrame, inplace: Optional[bool] = False) -> Union[pd.DataFrame, None]:
    """
    This helper function removes any invalid character, e.g. special characters and white spaces, in a column name and removes 
    leading characters until a character from a-z or A-Z is matched. Note this function does not replace python keywords or reserved 
    words if they exist as column names. Use the `rename()` method of pandas DataFrame or the datatable rename `{"A": "col_A"}` syntax 
    to clean the column names if `check_col_nms()` reveals such invalid columns names. 

    Parameters
    ----------
    df : Dataframe
    inplace : bool, optional
        Whether to return a new DataFrame, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with transformed column names or None if inplace=True.
    """
    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    original_col_nms = df.columns.tolist()
    # Remove leading characters until a letter is matched
    # ^ matches the beginning of the string
    # [^ ] is negated set, matching any character that is not in this set
    # + is a quantifier, matching the preceding element one or more times
    new_col_nms = (sub('^[^a-zA-Z]+', '', col) for col in original_col_nms)
    # Remove trailing characters until a letter is matched
    new_col_nms = (sub('[^a-zA-Z]+$', '', col) for col in new_col_nms)
    # Replace white spaces in-between words with "_"
    # \s+ matches 1 or more whitespace characters (spaces, tabs, line breaks)
    new_col_nms = (sub(r'\s+', '_', col) for col in new_col_nms)
    # [^a-zA-Z0-9_] matches any character that is not a 'word character' (alphanumeric & underscore), which is equivalent to \W
    df.columns = [sub('[^a-zA-Z0-9_]', '', col.lower()) for col in new_col_nms]

    # Return copy
    if (not inplace):
        return df
    else:
        return None


# ----------------------- Function for case conversion ----------------------- #


def case_convert(df: pd.DataFrame,
                 cols: Optional[Union[str, List[str]]] = None,
                 to: Optional[str] = 'lower',
                 inplace: Optional[bool] = False) -> Union[pd.DataFrame, None]:
    """
    This helper function converts the cases in the columns of a DataFrame; the aim is to address case inconsistencies 
    in string columns. Use this function to ensure values in a DataFrame are consistent with regards to case, which helps 
    reduce the chance of committing further errors later on in the cleaning process.

    Parameters
    ----------
    df : DataFrame
    cols : str or list of str, optional
        A single column name or a sequence of column names, by default None, which converts all columns that can be inferred as having 'string' dtypes.
    to : str, optional
        The direction or type of case conversion. One of 'lower', 'upper', 'title', or 'capitalize', by default 'lower'.
    inplace : bool, optional
        Whether to return a new DataFrame, by default False.

    Returns
    -------
    DataFrame
        A DataFrame with transformed columns or None if inplace=True.

    Raises
    ------
    TypeError
        The argument 'cols' must be registered as a Sequence or a single string.
    InvalidColumnDtypeError
        User supplied columns contain non-string columns.
    ValueError
        Direction or type of case conversion must either be 'lower', 'upper', 'title', or 'capitalize'.
    """
    if not is_list_str(cols) and cols is not None:
        raise TypeError(
            "'cols' must be a sequence like a list or a single string")

    # Create a copy if inplace=False
    if (not inplace):
        df = df.copy()

    # Branch 1: If user does not specify columns, default to using all columns that are inferred as 'string'
    if cols == None:
        bool_is_str = [is_string_dtype_pd(df[col])
                       for col in df.columns.tolist()]
        cols = list(compress(df.columns.tolist(), bool_is_str))
    # Branch 2: If user does specify, check input
    else:
        # Sub-branch 1: If 'cols' is a single string, then use the string to select column from df directly
        if isinstance(cols, str):
            bool_is_str = [is_string_dtype_pd(df[cols])]
        # Sub-branch 2: If 'cols' is a sequence of strings, then apply list comprehension
        else:
            bool_is_str = [is_string_dtype_pd(df[col]) for col in cols]

        # Take the 'bool_is_str' list from either one of the two sub-branches above, check input column data types
        if not all(bool_is_str):
            raise InvalidColumnDtypeError(col_nms=list(
                compress(cols, [not element for element in bool_is_str])), dtype='string')

    # Use pd.DataFrame since, if user passes a single str as 'cols', df[cols] would return a series, and lambda 'x' would be the elements
    # Apply relies on the Series 'str' attribute to work
    if (to == 'upper'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.upper())
    elif (to == 'lower'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.lower())
    elif (to == 'title'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.title())
    elif (to == 'capitalize'):
        df[cols] = pd.DataFrame(df[cols]).apply(lambda x: x.str.capitalize())
    else:
        raise ValueError(
            "'to' must either by 'lower', 'upper', 'title', or 'capitalize'")

    # Return copy
    if (not inplace):
        return df
    else:
        return None


# ---------------- Function to relocate columns in a DataFrame --------------- #


def relocate(df: pd.DataFrame,
             to_move: Union[str, List[str]],
             before: Union[str, None] = None,
             after: Union[str, None] = None) -> pd.DataFrame:
    """
    This function reorders the columns in a DataFrame based on a reference column, which is either 
    specified as `before` or `after`. Only one of 'before' and 'after' should be supplied as a string.

    Parameters
    ----------
    df : DataFrame
    to_move : str or list of str 
        A single column name or a list of column names to relocate.
    before : str or None, optional
        A reference column before which the columns in 'to_move' should be relocated, by default None.
    after : str or None, optional
        A reference column after which the columns in 'to_move' should be relocated, by default None.

    Returns
    -------
    DataFrame
        A DataFrame with reordered columns.

    Raises
    ------
    TypeError
        The argument 'to_move' must either be a list or a single string.
    TypeError
        Must supply only one of 'before' and 'after' as a string.
    """
    # Check input
    if not is_list_str(to_move):
        raise TypeError(
            "'to_move' must be a sequence like a list or a single string")
    # Exclusive or (exclusive disjunction where this expression is true if and only if the two booleans differ (one is true, the other is false))
    if isinstance(before, str) == isinstance(after, str):
        raise TypeError(
            "must supply only one of 'before' and 'after' as a string")

    # If 'to_move' is a string
    if isinstance(to_move, str):
        # Strings are immutable so this creates a new object that 'to_move' references in the local scope that is mutable
        to_move = [to_move]

    # If the reference column is included in 'to_move', remove it
    if before in to_move:
        to_move.remove(before)
    elif after in to_move:
        to_move.remove(after)

    # List of all column names
    cols = df.columns.tolist()

    # If we wish to move cols before the reference column
    if isinstance(before, str):
        # Select columns before the reference column (not including the reference column)
        seg1 = cols[:cols.index(before)]
        # Segment 2 includes columns to move 'before' the reference column (in other words, append 'before' to the end of 'to_move')
        to_move.append(before)
        seg2 = to_move

    # If we wish to move cols after the reference column
    if isinstance(after, str):
        # Select columns before the reference column (including the reference column)
        seg1 = cols[:cols.index(after) + 1]
        # Segment 2 simply includes columns to move
        seg2 = to_move

    # In either case, 'before' or 'after', we need to ensure columns to move are not in segment 1 (in other words, drop 'to_move' columns from their original positions)
    # Segment 2 in either case should always include columns to move
    seg1 = [col for col in seg1 if col not in seg2]
    # Finally, select the rest of the columns--- those that are not in seg1 and seg2
    seg3 = [col for col in cols if col not in seg1 + seg2]

    # For 'before', seg2 includes columns to move plus the reference column
    # Thus, seg2 created from the 'before' branch ensures that columns to move will appear 'before' the reference column
    # For 'after', adding seg1 (with reference column) + seg2 (to_move) in this order ensures that columns to move will appear 'after' the reference column
    # Finally, seg3 adds the rest of the columns to the end
    return df[seg1 + seg2 + seg3]


# ---------------------------------------------------------------------------- #
#                               Profiling helpers                              #
# ---------------------------------------------------------------------------- #

# ------------------ Function for identifying missing values ----------------- #


def find_missing(df: pd.DataFrame, axis: Optional[int] = 0) -> pd.Series:
    """
    This is a helper function that identifies columns or rows in a DataFrame
    that contain missing values. Then, rows and columns containing values may
    be retrieved using `df.loc[cu.find_missing(df, axis=1).index, :]` or 
    `df.loc[:, cu.find_missing(df, axis=0).index]` respectively.

    Parameters
    ----------
    df : DataFrame
    axis : int, optional
        Whether to identify rows or columns that contain missing values, by default 0 (columns).

    Returns
    -------
    Series of bool
        Boolean series indicating columns or rows with missing values.

    Raises
    ------
    TypeError
        The argument 'axis' must be an integer.
    ValueError
        The argument 'axis' must either be 1 (rows) or 0 (columns).
    """
    if not isinstance(axis, int):
        raise TypeError("The argument 'axis' must be an integer")

    # The lambda function simply returns the true elements of the boolean series
    if axis == 1:
        return df.isna().any(axis=1)[lambda x: x]
    elif axis == 0:
        return df.isna().any(axis=0)[lambda x: x]
    else:
        raise ValueError("'axis' must either be 1 (rows) or 0 (columns)")


# -------------- Function to create a tuple of frequency tables -------------- #


def freq_tbl(df: pd.DataFrame, dropna: Optional[bool] = False, cardinality: Optional[int] = 20, **kwargs: str) -> NamedTuple:
    """
    This function creates a sequence of freqency tables of the fields in a DataFrame, which can be examined to 
    identify misspellings or case inconsistencies. You may pass extra keyword arguments for the underlying pandas 
    function. See `?pandas.DataFrame.value_counts` for options (note that the `subset` argument is not permitted). 
    The function is `dtype` agnostic since the underlying function `pd.Series.value_counts` works on categorical and 
    categorical features.

    Parameters
    ----------
    df : DataFrame
    dropna : bool, optional
        Whether to drop missing values, by default False.
    cardinality : int, optional
        Cardinality limit for the fields in the input DataFrame for filtering out high cardinal fields, by default 20.

    Returns
    -------
    tuple of DataFrame
        A namedtuple of frequency tables containing counts (or proportions if extra keyword arguments are specified) per category for each categorical field.

    Raises
    ------
    ValueError
        Only 'normalize', 'sort', 'ascending' are supported as extra keyword arguments.
    TypeError
        The argument 'cardinality' must be an integer.
    """
    # Check keyword args
    if not all((kwarg in ('normalize', 'sort', 'ascending') for kwarg in kwargs.keys())):
        raise ValueError(
            "Only 'normalize', 'sort', and 'ascending' are supported as extra keyword arguments")
    # Check input
    if not isinstance(cardinality, int):
        raise TypeError("'cardinality' must be an integer")
    # Obtain number of unique values for each column in 'df'
    # Only use columns where the number of unique levels is less than or equal to the cardinality value (defaults to 20)
    df = df.loc[:, [col for col in df.columns if len(
        df[col].unique()) <= cardinality]]
    # Generator of frequency tables
    gen_of_freq = (pd.DataFrame(df[col].value_counts(dropna=dropna, **kwargs))
                   for col in df.columns)
    # Create subclass constructor
    freq = namedtuple('freq', df.columns)
    # Use constructor to create a named tuple
    freq_tbl = freq._make(gen_of_freq)

    return freq_tbl
