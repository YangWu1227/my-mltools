# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
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


@pytest.fixture(scope='module')
def test_data():
    return pd.read_csv(
        'tests/test_data.csv'
    )


@pytest.fixture(scope='module')
def invalid_data():
    return {
        'missing': pd.DataFrame({'longitude': (1, 2, np.NaN), 'latitude': (3, 4, 5)}),
        'object': pd.DataFrame({'longitude': (1, 'missing', 'missing2'), 'latitude': (3, 4, 5)})
    }


@pytest.fixture(scope='class')
def transformer_instance():
    return CoordinateTransformer()


class TestCoordinateTransformer:
    """
    Tests for the CoordinateTransformer custom transformer class.
    """

    def test_instantiate(self, transformer_instance):
        """
        Test that custom transformer has proper instantiation.
        """
        assert isinstance(transformer_instance, CoordinateTransformer)

        assert transformer_instance.get_params() == {
            'coord_cols': ['longitude', 'latitude'],
            'k_range': range(4, 13),
            'strategy': 'kmeans'
        }

    def test_fit(self, transformer_instance, test_data):
        """
        Test that the transformer can be fitted to data.
        """

        # Fit returns 'self'
        assert isinstance(transformer_instance.fit(
            test_data), CoordinateTransformer)

        # Check learned attributes
        assert hasattr(transformer_instance, 'optimal_k_')
        assert isinstance(transformer_instance.optimal_k_, np.int64)

        assert hasattr(transformer_instance, 'distortion_scores_')
        assert isinstance(transformer_instance.distortion_scores_, list)
        assert all([isinstance(score, float)
                   for score in transformer_instance.distortion_scores_])

    def test_transform(self, transformer_instance, test_data):
        """
        Test that the transformer transforms input data frame.
        """

        # Transform returns transformed data containing new 'cluster_label' column
        assert isinstance(transformer_instance.transform(
            test_data), pd.DataFrame)
        # Modifies in place
        assert 'kmean_cluster_labels' in test_data.columns.tolist()

        # Check attributes after 'transform'
        assert isinstance(transformer_instance.kmeans_, KMeans)
        assert isinstance(transformer_instance.labels_, np.ndarray)
        assert all([isinstance(label, np.int32)
                   for label in transformer_instance.labels_])

    def test_plot(self, transformer_instance, test_data, plt):
        """
        Test generated scatter plot.
        """
        transformer_instance.plot(
            test_data).saveas = "geospatial/test_transformer_scatter.png"

    def test_exceptions(self, invalid_data, test_data):
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
            fresh_instance.transform(test_data)

        # Calling 'plot' before 'transform'
        with pytest.raises(AttributeError, match="This 'CoordinateTransformer' instance is not transformed yet; please call 'transform' with appropriate arguments before plotting"):
            fresh_instance.plot(test_data)
