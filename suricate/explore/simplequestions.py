import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class SimpleQuestions(TransformerMixin):
    def __init__(self, n_questions=10):
        """

        Args:
            n_questions (int): number of lr_explore to be asked for each cluster
            isseries: if the input data is a serie or not
        """
        TransformerMixin.__init__(self)
        self.n_questions = n_questions
        self.n_clusters = None

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
        return self

    def transform(self, X):
        """
        Args:
            X (pd.Series/np.ndarray): Vector of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs

        Returns:
            np.ndarray: index number  of lines to take; dimension maximum is (n_clusters * n_questions, ) (some clusters may have a size inferior to n_questions)
        """
        if not isinstance(X, pd.Series):
            if X.ndim == 2:
                if X.shape[1] == 1:
                    X = X.flatten()
                else:
                    raise IndexError('Data must be 1-dimensionnal')
            y = pd.Series(data=X)
        else:
            y = X
        assert isinstance(y, pd.Series)

        questions = np.array([])
        for c in range(self.n_clusters):
            cluster = y.loc[y == c]
            if cluster.shape[0] < self.n_questions:
                sample_ix = cluster.index.values
            else:
                sample_ix = cluster.sample(self.n_questions).index.values
            questions = np.append(questions, sample_ix)
        questions = np.array(questions)
        return questions

    def predict(self, X):
        return self.transform(X)

    def fit_predict(self,X, y=None):
        return self.fit_transform(X, y=y)