"""Module comprising of the WellBehaviour Clustering Algorithm."""

import logging

import numpy as np
from kneed import KneeLocator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)


class EpsTuner(BaseEstimator, TransformerMixin):
    """Class to automatically select the eps parameter for DBSCAN(eps, min_samples).

    Instance Attributes:
    --------------------

        min_samples (int): Minimum number of data points, including the point itself, that are required in the eps neighbourhood
                           of a point for it to be considered as a core point.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.

        knn_distances (list): List of sorted KNN distances for kth = min_samples
        knee (kneed.KneeLocator): The fitted kneed.kneelocater model
        knee_idx (int): Index of the elbow in the knee plot
    """

    def __init__(self, min_samples):
        """Initialise the EpsTuner class."""
        super().__init__()

        # clustering model attributes
        self.min_samples = min_samples
        self.eps = None

        # attributes from the auto eps tuning
        self.knn_distances = None
        self.knee = None
        self.knee_idx = None

    def fit(self, X):
        """Perform the nearest neighbour distance calculations, find the knee (see references above).

        Keyword arguments:
        ------------------
        X -- The input data (features) for DBSCAN clustering.
        """
        nearest_neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors = nearest_neighbors.fit(X)
        distances, _indices = neighbors.kneighbors(X)
        distances = np.sort(distances[:, self.min_samples - 1], axis=0)
        i = np.arange(len(distances))
        # identify the knee in the KNN distances plot
        knee = KneeLocator(
            i,
            distances,
            S=1,
            curve="convex",
            direction="increasing",
            interp_method="polynomial",
        )
        # override the class attributes
        self.knn_distances = distances
        self.knee = knee
        self.knee_idx = knee.knee
        self.eps = distances[self.knee_idx]

        return self


class AutoDBSCAN(BaseEstimator, TransformerMixin):
    """Perform DBSCAN clustering with an automated, data-driven selection of the EPS parameter.

    Instance Attributes:

        min_samples (int): Minimum number of data points, including the point itself, that are required in the eps neighbourhood
                           of a point for it to be considered as a core point.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        clustering (sklearn.cluster.DBSCAN): The fitted DBSCAN model.

        knn_distances (list): List of sorted KNN distances for kth = min_samples
        knee (kneed.KneeLocator): The fitted kneed.kneelocater model
        knee_idx (int): Index of the elbow in the knee plot
    """

    def __init__(self, min_samples):
        """Instantiate the AutoDBSCAN class using the EpsTuner auto selected eps hyperparmeter.sign the min_samples variable."""
        super().__init__()

        # clustering model attributes
        self.min_samples = min_samples
        self.eps = None
        self.clustering = None

        # attributes from the auto eps tuning
        self.knn_distances = None
        self.knee = None
        self.knee_idx = None

    def fit(self, X):
        """
        Perform the DBSCAN clustering.

        Keyword arguments:
        ------------------
        X -- The input data (features) for DBSCAN clustering.
        """
        try:
            # instantiating the EpsTuner() class to auto select the epsilon parameter
            eps = EpsTuner(self.min_samples).fit(X)

            # storing the auto eps tuning information
            self.eps = eps.eps
            self.knn_distances = eps.knn_distances
            self.knee = eps.knee
            self.knee_idx = eps.knee_idx

            # assign the fitted DBSCAN model to the clustering attribute
            self.clustering = DBSCAN(eps=eps.eps, min_samples=self.min_samples).fit(X)

        except Exception as error:
            logging.error(error)

        return self
