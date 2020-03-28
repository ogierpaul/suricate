from sklearn.base import TransformerMixin, ClusterMixin, ClassifierMixin
import pandas as pd
from suricate.explore import SimpleQuestions, HardQuestions, ClusterClassifier, cluster_composition, KBinsCluster
from suricate.preutils import concatixnames

#TODO: Prio 1 Write Documentation

class Explorer(ClassifierMixin):
    def __init__(self, cluster=None, n_simple = 10, n_hard=10, ixname='ix', lsuffix='left', rsuffix='right'):
        """

        Args:
            cluster (ClusterMixin): if None, will use KbinsCluster with 25 clusters
        """
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        if cluster is None:
            cluster = KBinsCluster(n_clusters=25)
        self._cluster = cluster
        self._simplequestions = SimpleQuestions(n_questions=n_simple)
        self._hardquestions = HardQuestions(n_questions=n_hard)
        self._classifier = ClusterClassifier(ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix)
        pass

    def fit_cluster(self, X, y=None):
        """
        fit_cluster is to be called before fit
        Args:
            X (pd.DataFrame/np.ndarray): score matrix
            y: dummy

        Returns:
            np.ndarray
        """
        self._cluster.fit(X=X, y=y)
        return self._cluster


    def pred_cluster(self, X):
        """

        Args:
            X (np.ndarray): score matrix

        Returns:
            np.ndarray: 1-d vector of cluster
        """
        y_cluster = self._cluster.predict(X=X)
        return y_cluster


    def ask_simple(self, X, ix, fit_cluster=False):
        """

        Args:
            X (pd.DataFrame): score matrix with index
            ix (pd.Index): index of X

        Returns:
            pd.Index
        """
        if fit_cluster is True:
            self.fit_cluster(X=X)
        y_cluster = self.pred_cluster(X=X)
        y_cluster = pd.Series(data=y_cluster, name='y_cluster', index=ix)
        self._simplequestions.fit(X=y_cluster)
        ix_questions = self._simplequestions.transform(X=y_cluster)
        return ix_questions

    def ask_hard(self, X, y, ix, fit_cluster = False):
        """
        Args:
            X (np.ndarray): Score matrix
            y (pd.Series): y_true
            ix (pd.Index): index of X

        Returns:
            pd.Index: index of pointed questions
        """
        if fit_cluster is True:
            self.fit_cluster(X=X)
        y_cluster = self.pred_cluster(X=X)
        y_cluster = pd.Series(data=y_cluster, name='y_cluster', index=ix)
        self._hardquestions.fit(X=y_cluster, y=y)
        ix_questions = self._hardquestions.transform(X=y_cluster)
        return ix_questions

    def fit(self, X, y, fit_cluster = True):
        """

        Args:
            X (pd.DataFrame): score matrix with index
            y (pd.Series): labelled data (1 for a match, 0 if not), with index

        Returns:

        """
        if fit_cluster is True:
            self.fit_cluster(X=X)
        y_cluster = self._cluster.predict(X=X)
        y_cluster = pd.Series(data=y_cluster, index=X.index)
        self._classifier.fit(X=y_cluster, y=y)
        return self


    def predict(self, X):
        """

        Args:
            X (np.ndarray): score matrix

        Returns:
            np.ndarray: (0 for sure non matches, 1 for mixed matches, 2 for sure positive matches)
        """
        y_cluster = self.pred_cluster(X=X)
        y_pred = self._classifier.predict(X=y_cluster)
        return y_pred

    def transform(self, X):
        return self.predict(X=X)

    def fit_transform(self, X, y):
        self.fit(X=X, y=y, fit_cluster=True)
        return self.transform(X=X)

    def fit_predict(self, X, y):
        self.fit(X=X, y=y, fit_cluster=True)
        return self.predict(X=X)
