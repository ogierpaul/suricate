import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from suricate.explore.questions import _Questions

class SimpleQuestions(_Questions):
    def __init__(self, n_questions=10):
        """

        Args:
            n_questions (int): number of lr_explore to be asked for each cluster
        """
        _Questions.__init__(self, n_questions=n_questions)

    def fit(self, X, y=None):
        """
        Fit the transformer with the maximum number of clusters
        Args:
            X (pd.Series/np.ndarray): Vector of shape (n_pairs, 1) or (n_pairs,)  with the cluster classifications of the pairs
            y: vector

        Returns:
            self
        """
        if not (isinstance(X, pd.Series) or X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            raise IndexError('Expected dimension of array: ({a},1) or ({a},)'.format(a=X.shape[0]))
        self.n_clusters = np.unique(X).shape[0]
        self.clusters = np.unique(X)
        return self


    def predict(self, X):
        return self.transform(X)

    def fit_predict(self,X, y=None):
        return self.fit_transform(X, y=y)