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
from typing import List, Tuple, Union, Optional

# ------------------------------- Intra-package ------------------------------ #

from my_mltools.exceptions import (ColumnNameKeyWordError,
                                   ColumnNameStartWithDigitError,
                                   InvalidIdentifierError)

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
    # Remove trailing and leading white spaces
    new_col_nms = (col.strip() for col in original_col_nms)
    # Replace white spaces with "_"
    # \s+ matches 1 or more whitespace characters (spaces, tabs, line breaks)
    new_col_nms = (sub(r'\s+', '_', col) for col in new_col_nms)
    # [^a-zA-Z0-9_] matches any character that is not a 'word character' (alphanumeric & underscore), which is equivalent to \W
    new_col_nms = (sub('[^a-zA-Z0-9_]', '', col.lower())
                   for col in new_col_nms)
    # Remove leading characters until a letter is matched
    # ^ matches the beginning of the string
    # [^ ] is negated set, matching any character that is not in this set
    # + is a quantifier, matching the preceding element one or more times
    new_col_nms = (sub('^[^a-zA-Z]+', '', col) for col in new_col_nms)
    # Remove trailing characters until a letter is matched
    new_col_nms = [sub('[^a-zA-Z]+$', '', col) for col in new_col_nms]
    # Assign new columns
    df.columns = new_col_nms

    # Return copy
    if (not inplace):
        return df
    else:
        return None
