import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from suricate.preutils import concatixnames


class ClusterClassifier(ClassifierMixin):
    """
    This Classifier predicts for each cluster if the cluster is :
    - no match
    - all match
    - mixed match

    Input data (X) is a (n_pairs, ) pd.Series containing cluster values
    Fit data (y) is a (n_questions, ) pd.Series, with n_questions < n_pairs

    """
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', **kwargs):
        ClassifierMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        # clusters
        self.clusters = None

        # number of unique clusters
        self.n_clusters = None

        # clusters where no match has been found
        self.nomatch = None

        # clusters where all elements are positive matches
        self.allmatch = None

        # clusters where there is positive and negative values (matche and non-match)
        self.mixedmatch = None

        # Clusters not found (added in no matc)
        self.notfound = None

        self.fitted = False
        pass

    def fit(self, X, y):
        """
        For each cluster from X (y_cluster), use y_true (labelled data) to tell if the cluster contains:
        - only positive matches (self.allmatch)
        - only negative matches (self.nomatch)
        - positive and negative matches (self.mixedmatch)

        Cluster that are in X (y_cluster) and not in y (y_true) will be added to no_match

        Args:
            X (pd.Series): cluster vector from cluster.fit_predict() with index
            y (pd.Series): labelled data (1 for a match, 0 if not), with index

        Returns:

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
                self.clusters
            )
        )
        self.notfound = notfound
        if len(notfound) > 0:
            print('clusters {} are not found in y_true. they will be added to the no_match group'.format(notfound))
        self.nomatch += notfound

        self.fitted = True
        return self

    def predict(self, X):
        """
        This method returns for each cluster from X, an integer:
        - 0 if the cluster is a no-match cluster
        - 1 if the cluster is a mixed-match cluster
        - 2 if the cluster is an all-match cluster

        Args:
            X (np.array/pd.Series): 1-d array or Series, y_cluster

        Returns:
            np.ndarray: (0 for sure non matches, 1 for mixed matches, 2 for sure positive matches)
        """
        y_pred = np.isin(X, self.mixedmatch).astype(int) + 2 * np.isin(X, self.allmatch).astype(int)
        return y_pred

    def fit_predict(self, X, y):
        self.fit(X=X, y=y)
        y_pred = self.predict(X=X)
        return y_pred


def _check_ncluster_nquestions(n_questions, n_pairs, n_clusters):
    """

    Args:
        n_questions (int):
        n_clusters (int):
        n_pairs (int):

    Returns:
        boolean
    """
    if n_questions > n_pairs:
        return False
    elif n_questions * n_clusters > n_pairs:
        return False
    else:
        return True

def cluster_composition(y_cluster, y_true, normalize='index'):
    """

    Args:
        y_cluster (pd.Series): series with index
        y_true (pd.Series): series with index
        normalize:

    Returns:

    """
    ix_common = y_cluster.index.intersection(y_true.index)
    df = pd.crosstab(index=y_cluster.loc[ix_common], columns=y_true.loc[ix_common], normalize=normalize)
    df.sort_values(by=1, ascending=False, inplace=True)
    assert isinstance(df, pd.DataFrame)
    return df