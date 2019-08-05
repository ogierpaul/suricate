import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from suricate.lrdftransformers import cartesian_join
from suricate.preutils import concatixnames




class ClusterClassifier(ClassifierMixin):
    def __init__(self, cluster, ixname='ix', lsuffix='left', rsuffix='right', **kwargs):
        ClassifierMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        self.cluster = cluster
        self.n_clusters = None
        self.nomatch_clusters = None
        self.allmatch_clusters = None
        self.mixedmatch_clusters = None

    def fit(self, X, y):
        """

        Args:
            X (pd.DataFrame): X_score of n_samples, n_features with INDEX
            y: y_true

        Returns:

        """
        self.cluster.fit(X)
        self.n_clusters = self.cluster.n_clusters
        cluster_composition = self.cluster_composition(X=X, y=y)
        self.nomatch_clusters = cluster_composition.loc[
            (cluster_composition[1] == 0)
        ].index.tolist()
        self.allmatch_clusters = cluster_composition.loc[cluster_composition[0] == 0].index.tolist()
        self.mixedmatch_clusters = cluster_composition.loc[
            (cluster_composition[0] > 0) & (cluster_composition[1] > 0)
            ].index.tolist()
        self.fitted = True
        return self

    def predict(self, X):
        """

        Args:
            X (list): df_left, df_right

        Returns:
            np.ndarray (0 for sure non matches, 1 for mixed matches, 2 for sure positive matches)
        """
        y_cluster = self.cluster.predict(X=X)
        y_pred = np.isin(y_cluster, self.mixedmatch_clusters).astype(int) + 2 * np.isin(y_cluster,
                                                                                        self.allmatch_clusters).astype(
            int)
        return y_pred

    def fit_predict(self, X, y):
        self.fit(X=X, y=y)
        y_pred = self.predict(X=X)
        return y_pred

    def cluster_composition(self, X, y, normalize='index'):
        y_cluster = self.cluster.predict(X=X)
        cluster_composition = pd.crosstab(index=y_cluster, columns=y, normalize=normalize)
        return cluster_composition


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
