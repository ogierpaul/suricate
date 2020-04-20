import pandas as pd
import numpy as np
from sklearn.base import ClusterMixin
from sklearn.preprocessing import KBinsDiscretizer

class KBinsCluster(ClusterMixin):
    """
    This cluster transformer takes as input a similarity matrix X of size (n_samples, n_features).
    It then sums the score along the n_features axis, and discretize (i.e. Cluster) the scores using the KBinsDiscretizer
    From sklearn
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kb = KBinsDiscretizer(n_bins=n_clusters, strategy='uniform', encode='ordinal')
        self.fitted = False

    def fit(self, X, y=None):
        """
        Fit the KBinsDiscretizer
        Args:
            X (pd.DataFrame/np.ndarray): raw score input

        Returns:
            ClusterMixin
        """
        self.kb.fit(X=self._sumscore(X=X))
        self.fitted = True
        return self

    def _sumscore(self, X):
        """
        Sum the Score Matrix, and format it as a 2-dimensionnal array of dim (n_samples, 1)
        Args:
            X (np.ndarray/pd.DataFrame): score matrix

        Returns:
            np.ndarray: of dim (n_samples, 1)
        """
        return np.asarray(np.sum(X, axis=1)).reshape(-1, 1)


    def transform(self, X):
        """
        Returns the bin number
        Args:
            X:

        Returns:

        """
        return self.kb.transform(X=self._sumscore(X=X))

    def fit_predict(self, X, y=None):
        y_score = self._sumscore(X=X)
        self.kb.fit(X=y_score)
        return self.kb.transform(X=y_score)

    def fit_transform(self, X, y=None):
        y_score = self._sumscore(X=X)
        self.kb.fit(X=y_score)
        return self.kb.transform(X=y_score)

    def predict(self, X):
        return self.kb.transform(X=self._sumscore(X=X)).flatten()