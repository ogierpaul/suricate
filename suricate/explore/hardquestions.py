import numpy as np
import pandas as pd
from suricate.explore import cluster_composition
from suricate.explore.questions import _Questions


class HardQuestions(_Questions):
    def __init__(self, n_questions=10):
        _Questions.__init__(self, n_questions=n_questions)
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
            X (pd.Series): Matrix of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs
            y (pd.Series): series of correct answers, (n_answers, 2), where: \
                - index is a numerical value relative to X \
                - data is the classification (0 = not a match, 1: is a match) \

        Returns:
            self
        """
        # number of unique clusters
        self.n_clusters = np.unique(X).shape[0]

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
        if len(self.notfound) > 0:
            print('clusters {} are not found in y_true. they will be added to the no_match group'.format(notfound))
        self.nomatch += notfound

        self.fitted = True

        self.clusters = self.mixedmatch

        return self
