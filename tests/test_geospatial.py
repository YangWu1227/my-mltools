# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.cluster import KMeans
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError

# ----------------------------- Standard library ----------------------------- #

import os
from re import escape

# ------------------------------- Intra-package ------------------------------ #

from my_mltools.geospatial import CoordinateTransformer

# ---------------------------------------------------------------------------- #
#                     Tests for CoordinateTransformer class                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Test data -------------------------------- #


@pytest.fixture(scope='class')
def test_data():
    return pd.read_csv(
        'tests/test_data/coordinates.csv'
    )


@pytest.fixture(scope='class')
def test_data_ndarray(test_data):
    return test_data.to_numpy()


@pytest.fixture(scope='class')
def invalid_data():
    return {
        'missing': pd.DataFrame({'longitude': (1, 2, np.NaN), 'latitude': (3, 4, 5)}),
        'object': pd.DataFrame({'longitude': (1, 'missing', 'missing2'), 'latitude': (3, 4, 5)})
    }

# ------------------------------ Test instances ------------------------------ #


@pytest.fixture(scope='class')
def transformer_instance_pd():
    return CoordinateTransformer()


@pytest.fixture(scope='class')
def transformer_instance_ndarray():
    return CoordinateTransformer()

# -------------- Parametrize fixures for multiple runs of tests -------------- #


@pytest.mark.parametrize(
    'data, instance, plot_paths',
    [
        # For dataframe input
        (lazy_fixture('test_data'),
         lazy_fixture('transformer_instance_pd'),
         "geospatial/test_transformer_scatter_pd.png"),
        # For ndarray input
        (lazy_fixture('test_data_ndarray'),
         lazy_fixture('transformer_instance_ndarray'),
         "geospatial/test_transformer_scatter_ndarray.png")
    ],
    scope='class'
)
class TestCoordinateTransformer:
    """
    Tests for the CoordinateTransformer custom transformer class.
    """

    def test_instantiate(self, data, instance, plot_paths):
        """
        Test that custom transformer has proper instantiation.
        """
        assert isinstance(instance, CoordinateTransformer)

        assert instance.get_params() == {
            'k_range': range(4, 13),
            'strategy': 'kmeans'
        }

    def test_fit(self, data, instance, plot_paths):
        """
        Test that the transformer can be fitted to data.
        """

        # Fit returns 'self'
        assert isinstance(instance.fit(data),
                          CoordinateTransformer)

        # Check learned attributes
        assert hasattr(instance, 'optimal_k_')
        assert isinstance(instance.optimal_k_, np.int64)

        assert hasattr(instance, 'distortion_scores_')
        assert isinstance(instance.distortion_scores_, list)

        assert all([isinstance(score, float)
                   for score in instance.distortion_scores_])

    def test_transform(self, data, instance, plot_paths):
        """
        Test that the transformer transforms input data frame.
        """

        # Transform returns transformed 'cluster_label' ndarray
        assert isinstance(instance.transform(
            data), np.ndarray)

        # Check attributes after 'transform'
        assert isinstance(instance.X_coords, np.ndarray)
        assert isinstance(instance.kmeans_, KMeans)
        assert isinstance(instance.labels_, np.ndarray)
        assert all([isinstance(label, np.int32)
                   for label in instance.labels_])

    def test_plot(self, data, instance, plot_paths, plt):
        """
        Test generated scatter plot.
        """
        instance.plot().saveas = plot_paths

    def test_exceptions(self, invalid_data, data, instance, plot_paths):
        """
        Test exceptions raised.
        """

        fresh_instance = CoordinateTransformer()

        # Coordinate data contain missing
        with pytest.raises(ValueError, match=escape("Input contains NaN, infinity or a value too large for dtype('float64').")):
            fresh_instance.fit(invalid_data['missing'])

        # Coordinate data contain object dtype
        with pytest.raises(ValueError):
            fresh_instance.fit(invalid_data['object'])

        # Calling 'transform' before 'fit'
        with pytest.raises(NotFittedError, match="This CoordinateTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."):
            fresh_instance.transform(data)

        # Calling 'plot' before 'transform'
        with pytest.raises(AttributeError, match="This 'CoordinateTransformer' instance is not transformed yet; please call 'transform' with appropriate arguments before plotting"):
            fresh_instance.plot()
