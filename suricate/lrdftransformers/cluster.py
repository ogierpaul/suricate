import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from suricate.lrdftransformers import cartesian_join
from suricate.preutils import concatixnames


class ClusterClassifier(ClassifierMixin):
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
        self.n_clusters = None
        self.nomatch = None
        self.allmatch = None
        self.mixedmatch = None

    def fit(self, X, y):
        """

        Args:
            X (pd.Series): cluster vector from cluster.fit_predict() with index
            y: y_true

        Returns:

        """
        self.n_clusters = np.unique(X)
        cluster_composition = self.cluster_composition(y_cluster=X, y_true=y)
        self.nomatch = cluster_composition.loc[
            (cluster_composition[1] == 0)
        ].index.tolist()
        self.allmatch = cluster_composition.loc[cluster_composition[0] == 0].index.tolist()
        self.mixedmatch = cluster_composition.loc[
            (cluster_composition[0] > 0) & (cluster_composition[1] > 0)
            ].index.tolist()
        notfound = list(
            filter(
                lambda c: all(
                    map(
                        lambda m: c not in m,
                        [self.nomatch, self.allmatch, self.mixedmatch]
                    )
                ),
                self.n_clusters
            )
        )
        self.nomatch+=notfound
        self.fitted = True
        return self

    def predict(self, X):
        """

        Args:
            X (np.array/pd.Series): 1-d array or Series, y_cluster

        Returns:
            np.ndarray (0 for sure non matches, 1 for mixed matches, 2 for sure positive matches)
        """
        y_pred = np.isin(X, self.mixedmatch).astype(int) + 2 * np.isin(X, self.allmatch).astype(int)
        return y_pred

    def fit_predict(self, X, y):
        self.fit(X=X, y=y)
        y_pred = self.predict(X=X)
        return y_pred

    def cluster_composition(self, y_cluster, y_true, normalize='index'):
        """

        Args:
            y_cluster (pd.Series): series with index
            y_true (pd.Series): series with index
            normalize:

        Returns:

        """
        ix_common = y_cluster.index.intersection(y_true.index)
        df = pd.crosstab(index=y_cluster.loc[ix_common], columns=y_true.loc[ix_common], normalize=normalize)
        assert isinstance(df, pd.DataFrame)
        return df


def _return_cartesian_data(X, X_score, showcols, showscores, lsuffix, rsuffix, ixnamepairs):
    if showcols is None:
        showcols = X[0].columns.intersection(X[1].columns)
    X_data = cartesian_join(
        left=X[0][showcols],
        right=X[1][showcols],
        lsuffix=lsuffix,
        rsuffix=rsuffix
    ).set_index(ixnamepairs)
    mycols = list()
    for c in showcols:
        mycols.append(c + '_' + lsuffix)
        mycols.append(c + '_' + rsuffix)
    X_data = X_data[mycols]
    if showscores is not None:
        for c in showscores:
            X_data[c] = X_score[:, c]
    return X_data


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
