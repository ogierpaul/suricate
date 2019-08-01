import pandas as pd
from sklearn.base import ClusterMixin

from suricate.lrdftransformers.cluster import _return_cartesian_data
from suricate.preutils import concatixnames, createmultiindex


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
            ixname (str):
            lsuffix (str):
            rsuffix (str):
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
        # init the index
        self.ix = self._getindex(X=X)
        # Fit the transformer to create the scores
        self.transformer.fit(X=X)
        # Input the scores in a dataframe
        self.X_score = pd.DataFrame(
            data=self.transformer.transform(X=X),
            index=self.ix
        )
        # Fit the Cluster
        self.cluster.fit(X=self.X_score)
        self.n_clusters = self.cluster.n_clusters
        self.y_cluster = pd.Series(
            data=self.cluster.predict(X=self.X_score),
            index=self.ix,
            name='cluster'
        )
        # The first column of the score matrix is used for the similarity score
        self.score = self.X_score.iloc[:,0]
        self.score.name = 'similarity'

        # We then create a side_by_side dataframe of the input data
        self.X_sbs = _return_cartesian_data(
            X=X,
            X_score=self.X_score,
            showcols=self.showcols,
            showscores=self.showscore,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            ixnamepairs=self.ixnamepairs
        )
        # We add to this side_by_side the cluster and similarity score
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
            X: input matrix to be transformed by self.transformer ([df_left, df_right] at the moment)

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
            q_ix = q_ix.union(new_ix)
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