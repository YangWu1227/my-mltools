# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #

import numpy as np
import pandas as pd

# --------------------------------- Modeling --------------------------------- #

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# ------------------------ Elbow method for clustering ----------------------- #

from kneed import KneeLocator
from yellowbrick.cluster.elbow import distortion_score

# --------------------------------- Plotting --------------------------------- #

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

# ----------------------------- Standard library ----------------------------- #

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------- #
#                              Custom transformer                              #
# ---------------------------------------------------------------------------- #

class CoordinateTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling coordinate data. This transformer creates a column of cluster labels
    using the `sklearn.cluster.KMeans` learning algorithm that may be used in training in lieu of the original 
    coordinate data. Since this transformer accepts only a pandas DataFrame, it can only be used in the `pipeline`
    after an estimator whose `transform` or `fit_transform` returns a pandas DataFrame.


    Parameters
    ----------
    coord_cols : List[str], optional
        A list of columns containing coordinate data, by default ["longitude", "latitude"].
    strategy : str, optional
        The trategy for creating clustering labels, by default "kmeans".
    k_range : range, optional
        The range of value for the `n_clusters` parameter in `sklearn.cluster.KMeans` to try in order to determine an optimal value, by default range(4, 13).

    Attributes
    ----------
    distortion_scores_ : List[float]
        A list of mean distortions for each run of `sklearn.cluster.KMeans`. The distortion is computed as the the sum of the squared distances between each observation and 
        its closest centroid. Logically, this is the metric that K-Means attempts to minimize as it is fitting the model.
    optimal_k_: int
        The the optimal number of clusters.
    X_coords: ndarray of shape (n_samples, 2)
        The original coordinate data.
    kmeans_: object
        An instance of `sklearn.cluster.KMeans` fitted with the optimal number of clusters as its `n_clusters` parameter.
    labels_: ndarray of shape (n_samples,)
        The labels of each point.
    """

    def __init__(self, coord_cols: List[str] = ["longitude", "latitude"], strategy: str = "kmeans", k_range: range = range(4, 13)) -> None:
        self.coord_cols = coord_cols
        self.strategy = strategy
        self.k_range = k_range

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the transformer on X.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame.
        y : Optional[pd.Series], optional
            Ignored, present here for API consistency by convention, by default None.

        Returns
        -------
        self : object
            A fitted estimator.
        """
        # Input validate, convert coordinate columns to ndarray
        X_coords = check_array(
            X[self.coord_cols], accept_sparse=False, dtype="numeric")

        # Multiple runs
        distortion_scores = []
        for k in self.k_range:
            model = KMeans(n_clusters=k)
            model.fit_transform(X_coords)
            distortion_scores.append(distortion_score(X_coords, model.labels_))

        # Distortion scores
        self.distortion_scores_ = distortion_scores
        # Optimal number of clusters
        self.optimal_k_ = KneeLocator(
            x=self.k_range, y=self.distortion_scores_, curve='convex', direction='decreasing', S=1).knee

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input DataFrame by creating a column of cluster labels.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing a new column 'coord_cluster_label'.
        """
        # Check that the fit method has been called
        check_is_fitted(self, ('distortion_scores_', 'optimal_k_'))

        # Input validation
        X_coords = check_array(
            X[self.coord_cols], accept_sparse=False, dtype="numeric")

        # Coordinate columns
        self.X_coords = X_coords

        # Model
        self.kmeans_ = KMeans(n_clusters=self.optimal_k_, init='k-means++')

        # Store cluster labels as an attribute of the instance
        self.labels_ = self.kmeans_.fit_predict(X_coords)

        X['coord_cluster_label'] = self.labels_

        X.drop(columns=self.coord_cols, inplace=True)

        return X

    def plot(self) -> PathCollection:
        """
        Generate a scatter plot of coordinate points with marker colors representing
        the cluster to which each data point belongs. This method should only be called
        once the transformation has completed, i.e., after `transform` or `fit_transform`.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame.

        Returns
        -------
        PathCollection
            A collection of Paths, as created by `matplotlib.pyplot.scatter`.

        Raises
        ------
        AttributeError
            The 'labels_' field is created only after `transform` is called.
        """
        if not hasattr(self, 'labels_'):
            raise AttributeError(
                "This 'CoordinateTransformer' instance is not transformed yet; please call 'transform' with appropriate arguments before plotting")

        # Use numpy.take along columns of the ndarray
        plt.scatter(
            x=np.take(self.X_coords, indices=0, axis=1),
            y=np.take(self.X_coords, indices=1, axis=1),
            c=self.labels_, s=50, cmap='viridis'
        )

        plt.title('K-Means Clustered Map')
        plt.xlabel('Longitude')
        plt.ylabel("Latitude")

        return plt
