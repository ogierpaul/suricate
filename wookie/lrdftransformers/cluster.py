import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin, ClassifierMixin

from wookie.lrdftransformers import cartesian_join
from wookie.preutils import concatixnames, createmultiindex


# TODO: Save LrClusterQuestions X_score in order to gain a lot more time!!
# TODO: In ClusterClassifier, use PCA with 1 component, and a Normal Scaler to output the similarity score


class LrClusterQuestions(ClusterMixin):
    """
    Help visualize the scores
    Mix a transformer (FeatureUnion) and usecols data
    """

    def __init__(self, transformer, cluster, ixname='ix', lsuffix='left', rsuffix='right', **kwargs):
        """

        Args:
            transformer (FeatureUnion):
            cluster (ClusterMixin)
            ixname:
            lsuffix:
            rsuffix:
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
        # TODO: use outcol, transformer name

        try:
            self.scorecols = [c[1].outcol for c in self.transformer.transformer_list]
        except:
            try:
                self.scorecols = [c[0] for c in self.transformer.transformer_list]
            except:
                self.scorecols = None
        else:
            self.scorecols = None
        self.cluster = cluster
        self.n_clusters = None
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
        self.transformer.fit(X=X)
        X_score = self.transformer.transform(X=X)
        self.cluster.fit(X=X_score)
        self.n_clusters = self.cluster.n_clusters
        return self

    def fit_predict(self, X, y=None):
        self.fit(X=X)
        y_out = self.predict(X=X)
        return y_out

    def predict(self, X):
        X_score = self.transformer.transform(X=X)
        y_out = self.cluster.predict(X=X_score)
        return y_out

    def representative_questions(self, X, showcols=None, showscores=None, n_questions=20):
        """
        Similarity score is first column of transformer
        Args:
            X (list): [df_left, df_right]
            showcols (list): list of column names of both left and right dataframe to be put, default all
            showscores: position of colums in final scoring array to be shown

        Returns:

        """
        # TODO: check n_cluster and n_questions coherency
        # TODO: assert is fitted
        n_questions_c = int(n_questions / self.n_clusters)  # Number of questions per cluster
        n_questions_bonus = n_questions - self.n_clusters * n_questions_c  # Number of bonus questions

        X_score = self.transformer.transform(X=X)
        y_cluster = pd.Series(
            data=self.cluster.predict(X=X_score),
            index=self._getindex(X=X),
            name='cluster'
        )
        y_score = pd.Series(
            data=X_score[:, 0],
            index=self._getindex(X=X),
            name='similarityscore'
        )

        X_data = _return_cartesian_data(
            X=X,
            X_score=X_score,
            showcols=showcols,
            showscores=showscores,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            ixnamepairs=self.ixnamepairs
        )
        X_data['cluster'] = y_cluster
        X_data['similarity'] = y_score

        # start round of questionning
        q_ix = []
        for c in range(self.n_clusters):
            q_ix += y_cluster.loc[y_cluster == c].sample(n_questions_c).index.tolist()
        q_ix += X_data.sample(n_questions_bonus).index.tolist()
        return X_data.loc[q_ix]

    def pointed_questions(self, X, y, showcols=None, showscores=None, n_questions=20):
        X_score = self.transformer.transform(X=X)
        y_cluster = pd.Series(
            data=self.cluster.predict(X=X_score),
            index=self._getindex(X=X),
            name='cluster'
        )
        y_score = pd.Series(
            data=X_score[:, 0],
            index=self._getindex(X=X),
            name='similarity'
        )

        X_data = _return_cartesian_data(
            X=X,
            X_score=X_score,
            showcols=showcols,
            showscores=showscores,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            ixnamepairs=self.ixnamepairs
        )
        X_data['cluster'] = y_cluster
        X_data['similarity'] = y_score

        cluster_composition = self.cluster_composition(X=X, y=y)
        nomatch_clusters = cluster_composition.loc[
            (cluster_composition[1] == 0)
        ].index.tolist()
        allmatch_clusters = cluster_composition.loc[cluster_composition[0] == 0].index.tolist()
        mixed_clusters = cluster_composition.loc[
            (cluster_composition[0] > 0) & (cluster_composition[1] > 0)
            ].index.tolist()

        n_questions_c = int(n_questions / len(mixed_clusters))  # Number of questions per cluster
        n_questions_bonus = n_questions - len(mixed_clusters) * n_questions_c  # Number of bonus questions
        # start round of questionning
        q_ix = []
        for c in mixed_clusters:
            q_ix += y_cluster.loc[y_cluster == c].sample(n_questions_c).index.tolist()
        q_ix += X_data.loc[y_cluster.isin(mixed_clusters)].sample(n_questions_bonus).index.tolist()
        return X_data.loc[q_ix]

    def cluster_composition(self, X, y, normalize='index'):
        X_score = self.transformer.transform(X=X)
        y_cluster = pd.Series(
            data=self.cluster.predict(X=X_score),
            index=self._getindex(X=X),
            name='cluster'
        )
        cluster_composition = pd.crosstab(index=y_cluster.loc[y.index], columns=y, normalize=normalize)
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
