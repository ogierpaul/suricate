import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from suricate.explore import cluster_composition


class PointedQuestions(TransformerMixin):
    def __init__(self, n_questions=10):
        TransformerMixin.__init__(self)
        self.n_questions = n_questions
        # number of unique clusters
        self.n_clusters = None
        self.clusters = None

        # clusters where no match has been found
        self.nomatch = None

        # clusters where all elements are positive matches
        self.allmatch = None

        # clusters where there is positive and negative values (matche and non-match)
        self.mixedmatch = None

        # Clusters not found (added in no matc)
        self.notfound = None

        self.fitted = False

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
        # number of unique clusters
        self.n_clusters = np.unique(X).shape[0]
        self.clusters = np.unique(X)

        # clusters composition, how many matches have been found, from y_true (supervised data)
        df_cluster_composition = cluster_composition(y_cluster=X, y_true=y)

        # clusters where no match has been found
        self.nomatch = df_cluster_composition.loc[
            (df_cluster_composition[1] == 0)
        ].index.tolist()

        # clusters where all elements are positive matches
        self.allmatch = df_cluster_composition.loc[df_cluster_composition[0] == 0].index.tolist()

        # clusters where there is positive and negative values (matche and non-match)
        self.mixedmatch = df_cluster_composition.loc[
            (df_cluster_composition[0] > 0) & (df_cluster_composition[1] > 0)
            ].index.tolist()

        # clusters that do not appear on y_true will be added to no match
        notfound = list(
            filter(
                lambda c: all(
                    map(
                        lambda m: c not in m,
                        [self.nomatch, self.allmatch, self.mixedmatch]
                    )
                ),
                range(self.n_clusters)
            )
        )
        self.notfound = notfound
        print('clusters {} are not found in y_true. they will be added to the no_match group'.format(notfound))
        self.nomatch += notfound

        self.fitted = True
        return self

    def transform(self, X):
        """
        Args:
            X (np.ndarray): Matrix of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs

        Returns:
            np.ndarray: index number (np) of lines to take
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
        for c in self.mixedmatch:
            cluster = y.loc[y == c]
            if cluster.shape[0] < self.n_questions:
                sample_ix = cluster.index.values
            else:
                sample_ix = cluster.sample(self.n_questions).index.values
            questions = np.append(questions, sample_ix)
        questions = np.array(questions)
        return questions
