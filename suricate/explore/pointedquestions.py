import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class PointedQuestions(TransformerMixin):
    def __init__(self, n_questions=10):
        TransformerMixin.__init__(self)
        self.n_questions = n_questions
        self.n_clusters = None

    def fit(self, X, y):
        """
        Fit the transformer with the maximum number of clusters
        Args:
            X (np.ndarray): Matrix of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs
            y (pd.Series): series of correct answers, (n_answers, 2), where: \
                - index is a numerical value relative to X \
                - data is the classification (0 = not a match, 1: is a match) \

        Returns:
            self
        """
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            raise IndexError('Expected dimension of array X: ({a},1) or ({a},)'.format(a=X.shape[0]))
        # if not (y.ndim == 2 and y.shape[1] == 2):
        if not isinstance(y, pd.Series):
            raise IndexError('Expected dimension of array y: ({a}, 2) '.format(a=y.shape[0]))
        self.n_clusters = np.max(X) + 1
        return self

    def transform(self, X):
        """
        Args:
            X (np.ndarray): Matrix of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs

        Returns:
            np.ndarray: index number (np) of lines to take
        """
        y = pd.Series(X)
        questions = []
        for c in range(self.n_clusters):
            questions += y.loc[y == c].sample(5).index.tolist()
        questions = np.array(questions)
        return questions

    def _cluster_composition(self, X, y, normalize='index'):
        """

        Args:
            X (numpy.ndarray): cluster vector
            y (pd.Series): labelized data
            normalize: normalize parameter of pd.crosstab

        Returns:
            pd.DataFrame: cross-tab analysis of y vers y_cluster composition of the data
        """
        y_cluster = pd.Series(X)
        y_true = y
        commonindex = y_true.index.intersection(y_cluster.index)
        cluster_composition = pd.crosstab(
            index=self.y_cluster.loc[commonindex],
            columns=y.loc[commonindex],
            normalize=normalize
        )
        return cluster_composition

    def mixed_clusters(self, cluster_composition):
        nomatch_clusters = cluster_composition.loc[
            (cluster_composition[1] == 0)
        ].index.tolist()
        allmatch_clusters = cluster_composition.loc[cluster_composition[0] == 0].index.tolist()
        mixed_clusters = cluster_composition.loc[
            (cluster_composition[0] > 0) & (cluster_composition[1] > 0)
            ].index.tolist()
        # TODO: Stop here.  Pointed question should be fed from a y_cluster_predictor giving nomatch_cluster, allmatch_clusters, mixed_clusters