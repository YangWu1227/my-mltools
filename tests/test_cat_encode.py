# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.exceptions import NotFittedError

# ----------------------------- Standard library ----------------------------- #

import os
from re import escape

# ------------------------------- Intra-package ------------------------------ #

from my_mltools.cat_encode import Embedder


# ---------------------------------------------------------------------------- #
#                           Tests for Embedder class                           #
# ---------------------------------------------------------------------------- #

# --------------------------------- Test data -------------------------------- #

@pytest.fixture(scope='class')
def test_data():
    return pd.read_csv(
        'tests/test_data/geospatial_test_data.csv',
        usecols=['ocean_proximity']
    )


@pytest.fixture(scope='class')
def test_data_ndarray(test_data):
    return test_data.to_numpy()


@pytest.fixture(scope='class')
def invalid_data():
    return pd.Series(('cat2', 'cat1', 'cat2', 'cat1', np.NaN))

# ------------------------------ Test instances ------------------------------ #


@pytest.fixture(scope='class')
def transformer_instance_series():
    return Embedder(key='ocean', num_oov_buckets=4, dimension=4)


@pytest.fixture(scope='class')
def transformer_instance_ndarray():
    return Embedder(key='ocean', num_oov_buckets=4, dimension=4)

# -------------- Parametrize fixures for multiple runs of tests -------------- #


@pytest.mark.parametrize(
    'data, instance',
    [
        # For dataframe input
        (lazy_fixture('test_data'),
         lazy_fixture('transformer_instance_series')),
        # For ndarray input
        (lazy_fixture('test_data_ndarray'),
         lazy_fixture('transformer_instance_ndarray'))
    ],
    scope='class'
)
class TestEmbedder:
    """
    Tests for the Embedder custom transformer class.
    """

    def test_instantiate(self, data, instance):
        """
        Test that custom transformer has proper instantiation.
        """
        assert isinstance(instance, Embedder)

        assert instance.get_params() == {
            'dimension': 4,
            'key': 'ocean',
            'num_oov_buckets': 4
        }

    def test_fit(self, data, instance):
        """
        Test that the transformer can be fitted to data.
        """

        # Fit returns 'self'
        assert isinstance(instance.fit(data), Embedder)

        # Check learned attributes
        assert hasattr(instance, 'vocab_')
        assert isinstance(instance.vocab_, np.ndarray)

        # Should be the same as calling np.unique manually on the data
        assert all(instance.vocab_ == np.unique(data))

    def test_transform(self, data, instance):
        """
        Test that the transformer transforms input data.
        """

        # Transform returns transformed 'values_' ndarray
        assert isinstance(instance.transform(data), np.ndarray)

        # Check attributes after 'transform'
        assert isinstance(instance.values_, np.ndarray)
        # Shape should match the number of rows of the input data and 'self.dimension'
        assert instance.values_.shape == (data.shape[0], instance.dimension)
        # Data type should be 'float32'
        assert instance.values_.dtype == np.float32

    def test_exceptions(self, invalid_data, data, instance):
        """
        Test exceptions raised.
        """

        fresh_instance = Embedder()

        # Data contain missing
        with pytest.raises(ValueError, match=escape("Input contains NaN")):
            fresh_instance.fit(invalid_data)

        # Calling 'transform' before 'fit'
        with pytest.raises(NotFittedError, match="This Embedder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."):
            fresh_instance.transform(data)
