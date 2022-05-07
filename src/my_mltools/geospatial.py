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

from matplotlib.pyplot import scatter
from matplotlib.collections import PathCollection

# ----------------------------- Standard library ----------------------------- #

from typing import List, Optional


# ---------------------------------------------------------------------------- #
#                              Custom transformer                              #
# ---------------------------------------------------------------------------- #

class CoordinateTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling coordinate data. This transformer creates a column of cluster labels
    using the `sklearn.cluster.KMeans` learning algorithm that may be used in training in lieu of the original 
    coordinate data.


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
            A pandas DataFrame containing a new column 'kmean_cluster_labels'.
        """
        # Check that the fit method has been called
        check_is_fitted(self, ('distortion_scores_', 'optimal_k_'))

        # Input validation
        X_coords = check_array(
            X[self.coord_cols], accept_sparse=False, dtype="numeric")

        # Model
        self.kmeans_ = KMeans(n_clusters=self.optimal_k_, init='k-means++')

        # Store cluster labels as an attribute of the instance
        self.labels_ = self.kmeans_.fit_predict(X_coords)

        X['kmean_cluster_labels'] = self.labels_

        return X

    def plot(self, X: pd.DataFrame) -> PathCollection:
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

        return scatter(x=X[self.coord_cols[0]], y=X[self.coord_cols[1]],
                       c=self.labels_, s=50, cmap='viridis')
