import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class _Questions(TransformerMixin):
    def __init__(self, n_questions=10):
        """

        Args:
            n_questions (int): number of lr_explore to be asked for each cluster
            isseries: if the input data is a serie or not
        """
        TransformerMixin.__init__(self)
        self.n_questions = n_questions
        self.n_clusters = None
        self.clusters = None

    def transform(self, X):
        """
        Args:
            X (pd.Series): Vector of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs

        Returns:
            pd.Index: index number  of lines to take; dimension maximum is (n_clusters * n_questions, ) (some clusters may have a size inferior to n_questions)
        """
        if not isinstance(X, pd.Series):
            raise IndexError('Data must be a 1-dimensionnal pandas series with index')

        questions = pd.MultiIndex(levels=[[],[]],
                             codes=[[],[]],
                             names=X.index.names)
        for c in self.clusters:
            cluster = X.loc[X == c]
            if cluster.shape[0] < self.n_questions:
                sample_ix = cluster.index
            else:
                sample_ix = cluster.sample(self.n_questions).index
            questions = questions.union(sample_ix)
        return questions

    def predict(self, X):
        return self.transform(X)

    def fit_predict(self,X, y=None):
        return self.fit_transform(X, y=y)