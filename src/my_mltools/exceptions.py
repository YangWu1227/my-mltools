# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

# ----------------------------- Standard library ----------------------------- #

from typing import List

# ---------------------------------------------------------------------------- #
#                       Cleaning utils module exceptions                       #
# ---------------------------------------------------------------------------- #

# ------------------------------- Keyword error ------------------------------ #


class ColumnNameKeyWordError(Exception):
    """
    Exception raised when column names are keywords.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'Columns {self.col_nms} are keywords of the language, and cannot be used as ordinary identifiers'

# -------------------------- Start with digit error -------------------------- #


class ColumnNameStartWithDigitError(Exception):
    """
    Exception raised when column names start with digits.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'Columns {self.col_nms} must not start with digits'

# ------------------------ Catch-all identifier error ------------------------ #


class InvalidIdentifierError(Exception):
    """
    Exception raised when column names are invalid identifiers.
    """

    def __init__(self, col_nms: List[str]) -> None:
        self.col_nms = col_nms

    def __str__(self) -> str:
        return f'Columns {self.col_nms} are invalid identifiers'
