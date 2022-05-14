# ---------------------------------------------------------------------------- #
#                           Load packages and modules                          #
# ---------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

# ------------------------------- Transformers ------------------------------- #

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# -------------------------------- Tensorflow -------------------------------- #

import tensorflow as tf
from tensorflow import keras

# ----------------------------- Standard library ----------------------------- #

from typing import Union, Dict, Iterable


# ---------------------------------------------------------------------------- #
#                              Custom transformer                              #
# ---------------------------------------------------------------------------- #

class Embedder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling high cardinal categorical data, i.e., factors with potentially many unique levels. 
    This transformer creates `dimension` number of numerical columns from an input categorical column using `keras`'s 
    `feature_column` module. This is a form of distributed representation of categorical data but in relatively lower 
    dimensional space compared to a full-blown sparse matrix (one-hot encoding) representation of all the unique categories.


    Parameters
    ----------
    key : str, optional
        A unique string identifying the input feature. It is used as the column name and the dictionary key for the feature columns, by default 'feature_key'.
    num_oov_buckets : int, optional
        A non-negative integer allocating the number of out-of-vocabulary buckets for potentially unknown categories in unseen data. All out-of-vocabulary inputs 
        will be assigned IDs in the range `[len(unique_categories), len(unique_categories) + num_oov_buckets)` based on a hash of the input value.
    dimension : int, optional
        An integer specifying the dimension of the embedding, by default 2 and must be > 0. This parameter can be tuned via a grid search when used in conjunction 
        with a preprocessing pipeline.

    Attributes
    ----------
    vocab_ : ndarray
        The sorted unique values of the input categorical feature.
    values_ : ndarray, shape (n_samples, dimension)
        The numerical columns obtained from embedding.
    """

    def __init__(self, key: str = 'feature_key', num_oov_buckets: int = 2, dimension: int = 2) -> None:
        self.key = key
        self.num_oov_buckets = num_oov_buckets
        self.dimension = dimension

    def fit(self, X: Union[pd.Series, np.ndarray], y=None):
        """
        Fit the transformer on X.

        Parameters
        ----------
        X : Union[pd.Series, np.ndarray]
            One dimensional column vector representing the categorical feature.
        y : optional
            Ignored, present here for API consistency by convention, by default None.

        Returns
        -------
        self : object
            A fitted estimator.
        """

        self.vocab_ = np.unique(check_array(X, dtype=None, ensure_2d=False))

        return self

    def __call_feature_columns(self, feature_column: Iterable, inputs: Dict) -> tf.Tensor:
        """
        A convenient way to call a 'CategoricalColumn' to display its output.

        Parameters
        ----------
        feature_column : Iterable
            An iterable containing the FeatureColumns to use as inputs to the model. All items should be instances of classes derived from `DenseColumn`.
        inputs : Dict
            A model's inputs are always expected to be a dictionary of tensors.

        Returns
        -------
        Tensor
            A dense Tensor produced based on the given `feature_columns`.
        """
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        return feature_layer(inputs)

    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform input categorical column into `dimension` number of numerical column vectors.

        Parameters
        ----------
        X : Union[pd.Series, np.ndarray]
            One dimensional column vector representing the categorical feature.

        Returns
        -------
        X_transformed : np.ndarray,  shape (n_samples, dimension)
            An ndarray of numerical columns.
        """
        # Check that the fit method has been called
        check_is_fitted(self, 'vocab_')

        X_vector = check_array(
            X, dtype=None, ensure_2d=False).reshape(-1).tolist()

        # Tensorflow CategoricalColumn
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key=self.key,
            vocabulary_list=self.vocab_,
            num_oov_buckets=self.num_oov_buckets
        )

        # Tensorflow DenseColumn
        dense_column = tf.feature_column.embedding_column(
            cat_column, dimension=self.dimension)

        # Store values
        self.values_ = self.__call_feature_columns(
            dense_column, {self.key: X_vector}).numpy()

        return self.values_
