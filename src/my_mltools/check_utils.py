# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

from pandas.api.types import is_string_dtype
from pandas import Series

# ----------------------------- Standard library ----------------------------- #

from collections.abc import Sequence
from typing import List

# ---------------------------------------------------------------------------- #
#                    Helper predicates for input validation                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Builtins --------------------------------- #


def is_list_str(seq: Sequence) -> bool:
    """
    This helper returns `True` if the input is either a list or a string.

    Parameters
    ----------
    seq : Sequence of objects
        An input sequence to be tested.

    Returns
    -------
    bool
        `True` if the sequence is either a list or a string..
    """
    return isinstance(seq, list) or isinstance(seq, str)

# ----------------------------- Helper for pandas ---------------------------- #


def is_string_dtype_pd(col: Series) -> bool:
    """
    This helper checks whether the provided Series is of the string dtype.

    Parameters
    ----------
    col : Series
        A pandas Series to be tested.

    Returns
    -------
    bool
        `True` if the Series is of the string dtype.
    """
    return is_string_dtype(col)
