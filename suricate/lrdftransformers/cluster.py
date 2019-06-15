import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, ClassifierMixin

from suricate.lrdftransformers import cartesian_join
from suricate.preutils import concatixnames, createmultiindex


# TODO: In ClusterClassifier, use PCA with 1 component, and a Normal Scaler to output the similarity score


class ClusterQuestions(ClusterMixin):
    """
    Help visualize the scores
    Mix a transformer (FeatureUnion) and usecols data
    """

    def __init__(self, transformer, cluster, ixname='ix', lsuffix='left', rsuffix='right',
                 showcols=None, showscore=None,
                 **kwargs):
        """

        Args:
            transformer (FeatureUnion):
            cluster (ClusterMixin)
            ixname:
            lsuffix:
            rsuffix:
            showcols (list): list of column names of both left and right dataframe to be put, default all
            showscores: position of colums in final scoring array to be shown
            **kwargs:
        """
        ClusterMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        self.transformer = transformer
        self.showcols = showcols
        self.showscore = showscore

        try:
            self.scorecols = [c[1].outcol for c in self.transformer.transformer_list]
        except:
            try:
                self.scorecols = [c[0] for c in self.transformer.transformer_list]
            except:
                self.scorecols = None
        else:
            self.scorecols = None
        # Boolean to check if it is fitted
        self.fitted = False
        # Cluster MiXin
        self.cluster = cluster
        # Int, number of clusters
        self.n_clusters = None
        # Score matrix (n_samples, n_features)
        self.X_score = pd.DataFrame()
        # Vector of cluster shape = (n_samples, 1)
        self.y_cluster = pd.Series()
        # Index of X_score of length (n_samples)
        self.ix = pd.Index([])
        self.X_sbs = pd.DataFrame()
        pass

    def _getindex(self, X):
        """
        Return the cartesian product index of both dataframes
        Args:
            X (list): [df_left, df_right]
            y (pd.Series/pd.DataFrame/pd.MultiIndex): dummy, not used

        Returns:
            pd.MultiIndex
        """
        ix = createmultiindex(X=X, names=self.ixnamepairs)
        return ix

    def fit(self, X=None, y=None):
        """

        Args:
            X: input matrix to be transformed by self.transformer
            y: dummy

        Returns:
            ClusterMixin
        """
        self.ix = self._getindex(X=X)
        self.transformer.fit(X=X)
        self.X_score = pd.DataFrame(
            data=self.transformer.transform(X=X),
            index=self.ix
        )
        self.cluster.fit(X=self.X_score)
        self.n_clusters = self.cluster.n_clusters
        self.y_cluster = pd.Series(
            data=self.cluster.predict(X=self.X_score),
            index=self.ix,
            name='cluster'
        )
        self.score = self.X_score.iloc[:,0]
        self.score.name = 'similarity'

        self.X_sbs = _return_cartesian_data(
            X=X,
            X_score=self.X_score,
            showcols=self.showcols,
            showscores=self.showscore,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            ixnamepairs=self.ixnamepairs
        )
        self.X_sbs['cluster'] = self.y_cluster
        self.X_sbs['similarity'] = self.score
        self.fitted = True
        return self

    def fit_predict(self, X, y=None):
        """

        Args:
            X (list): (df_left, df_right]
            y (np.ndarray): clusters

        Returns:

        """
        self.fit(X=X)
        y_out = self.predict(X=X)
        return y_out

    def predict(self, X):
        """

        Args:
            X: input matrix to be transformed by self.transformer

        Returns:
            np.ndarray: prediction of cluster
        """
        X_score = self.transformer.transform(X=X)
        y_out = self.cluster.predict(X=X_score)
        return y_out


    def representative_questions(self, n_questions=20):
        """
        Similarity score is first column of transformer
        Args:
            X (list): [df_left, df_right]

            n_questions: number of questions per cluster

        Returns:
            pd.DataFrame: side by side pairs
        """
        assert self.fitted is True

        # start round of questionning
        q_ix = self._findpairs(on_ix=self.ix, n_questions=n_questions, on_cluster=range(self.n_clusters))
        return self.X_sbs.loc[q_ix]

    def _findpairs(self, on_ix,  n_questions, on_cluster=None):
        """
        Find n_questions for each cluster in on_ix
        If n_questions > len(cluster) for a given cluster, we take the full cluster
        Args:
            on_ix (pd.Index): index on which to find the questions
            n_questions (int): number of questions per cluster
            on_cluster (list): list of clusters

        Returns:
            pd.Index
        """
        q_ix = pd.Index([])
        for c in on_cluster:
            d = self.y_cluster.loc[
                on_ix
            ].loc[
                self.y_cluster.loc[on_ix] == c
            ]
            if d.shape[0] < n_questions:
                print('n_questions bigger than size of cluster for cluster {}'.format(c))
                new_ix = d.index
            else:
                new_ix = d.sample(
                    n_questions
                ).index
            q_ix.union(new_ix)
        return q_ix

    def pointed_questions(self, y, n_questions=20):
        """

        Args:
            y (pd.Series): labelled series (0 if pair is not amatch, 1 otherwise), index (['ix_left', 'ix_right])
            n_questions (int): number of questions per cluster

        Returns:
            pd.DataFrame: side by side pairs
        """

        cluster_composition = self.cluster_composition(y=y)
        nomatch_clusters = cluster_composition.loc[
            (cluster_composition[1] == 0)
        ].index.tolist()
        allmatch_clusters = cluster_composition.loc[cluster_composition[0] == 0].index.tolist()
        mixed_clusters = cluster_composition.loc[
            (cluster_composition[0] > 0) & (cluster_composition[1] > 0)
            ].index.tolist()

        on_ix = self.y_cluster.loc[
            self.y_cluster.isin(mixed_clusters)
        ].index.difference(
            y.index
        )

        # start round of questionning
        q_ix = self._findpairs(on_ix=on_ix, on_cluster=mixed_clusters, n_questions=n_questions)

        return self.X_sbs.loc[q_ix]

    def cluster_composition(self, y, normalize='index'):
        """

        Args:
            y (pd.Series): labelized data
            normalize: normalize parameter of pd.crosstab

        Returns:
            pd.DataFrame: cross-tab analysis of y vers y_cluster composition of the data
        """
        commonindex = y.index.intersection(self.y_cluster.index)
        cluster_composition = pd.crosstab(
            index=self.y_cluster.loc[commonindex],
            columns=y.loc[commonindex],
            normalize=normalize
        )
        return cluster_composition


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
            y:

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
