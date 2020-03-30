from sklearn.base import TransformerMixin, ClusterMixin, ClassifierMixin
import pandas as pd
from suricate.explore import SimpleQuestions, HardQuestions, ClusterClassifier, cluster_composition, KBinsCluster
from suricate.preutils import concatixnames


class Explorer(ClassifierMixin):
    """
    Most important piece of the explorer module.
    The Explorer class is a classifier that:
    * Using the cluster class provided, cluster the data according to the similarity matrix provided
    * Using the SimpleQuestions and HardQuestions classes, generate indexes of pairs needed
    * Using the ClusterClassifier class, classify the input data as a match or potentially not a match
    """
    def __init__(self, clustermixin=None, n_simple = 10, n_hard=10, ixname='ix', lsuffix='left', rsuffix='right'):
        """

        Args:
            clustermixin (ClusterMixin): if None, will use KbinsCluster with 25 clusters
            n_simple (int): number of simple questions per cluster
            n_hard (int): number of hard questions per cluster
            ixname (str): default 'ix'
            lsuffix (str): default 'left'
            rsuffix (str): default 'right'
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
        if clustermixin is None:
            clustermixin = KBinsCluster(n_clusters=25)
        self._clustermixin = clustermixin
        self._simplequestions = SimpleQuestions(n_questions=n_simple)
        self._hardquestions = HardQuestions(n_questions=n_hard)
        self._clusterclassifier = ClusterClassifier(ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix)
        pass

    def fit_cluster(self, X, y=None):
        """
        Fit the clustermixin.to be called before fit. Use only X, y is dummy (as ClusterMixIn fit is non-supervized).
        Args:
            X (pd.DataFrame/np.ndarray): score matrix
            y: dummy value to respect the convention of fit(X,y). Not used in non-supervised learning.

        Returns:
            np.ndarray
        """
        self._clustermixin.fit(X=X, y=y)
        return self._clustermixin


    def pred_cluster(self, X):
        """
        Predict the cluster number of the row in the similarity matri, using the ClusterMixin Transformer
        Args:
            X (np.ndarray): score matrix

        Returns:
            np.ndarray: 1-d vector of cluster
        """
        y_cluster = self._clustermixin.predict(X=X)
        return y_cluster


    def ask_simple(self, X, fit_cluster=False):
        """
        From the similarity matrix,
        generate for each cluster a number of sample pairs (questions).
        If fit_cluster is True, fits the ClusterMixIn on X first.
        Args:
            X (pd.DataFrame): score matrix with index
            ix (pd.Index): index of X. Important to generate the index of the questions.
            fit_cluster (bool): Fit the clustermixin on X, default False

        Returns:
            pd.Index
        """
        if fit_cluster is True:
            self.fit_cluster(X=X)
        y_cluster = self.pred_cluster(X=X)
        y_cluster = pd.Series(data=y_cluster, name='y_cluster', index=X.index)
        self._simplequestions.fit(X=y_cluster)
        ix_questions = self._simplequestions._transform(X=y_cluster)
        return ix_questions

    def ask_hard(self, X, y, fit_cluster = False):
        """
        From the similarity matrix,
        generate for each mixed_match cluster a number of sample pairs (questions).
        If fit_cluster is True, fits the ClusterMixIn on X first.
        Using y, determine the mixed_match clusters
        Args:
            X (pd.DataFrame): Score matrix, with index. Index is important to match X with labellized pairs y, and to give the indexes of hard questions.
            y (pd.Series): y_true. Important to label the pairs and find the mixed_match questions
            fit_cluster (bool): Fit the clustermixin on X, default False

        Returns:
            pd.Index: index of pointed questions
        """
        if fit_cluster is True:
            self.fit_cluster(X=X)
        y_cluster = pd.Series(data=self.pred_cluster(X=X), name='y_cluster', index=X.index)
        self._hardquestions.fit(X=y_cluster, y=y)
        ix_questions = self._hardquestions._transform(X=y_cluster)
        return ix_questions

    def fit(self, X, y, fit_cluster = True):
        """
        * If fit_cluster is True, fits the clustermixin on X
        * Predict y_cluster from X using the clustermixin
        * Using labellized data y (y_true), fit the cluster classifier: Determine nomatch, allmatch, and mixedmatch clusters
        * in This fit, X is a pd.DataFrame / not an array
        Args:
            X (pd.DataFrame): score matrix with index
            y (pd.Series): labelled data (1 for a match, 0 if not), with index
            fit_cluster (bool): If false, will only fit the clusterclassifier. If true, will also fit the clustermixin. Default True

        Returns:

        """
        if fit_cluster is True:
            self.fit_cluster(X=X)
        y_cluster = pd.Series(data=self._clustermixin.predict(X=X), index=X.index)
        self._clusterclassifier.fit(X=y_cluster, y=y)
        return self


    def predict(self, X):
        """
        This method returns for each row from X, an integer:
        - 0 if the cluster is a no-match cluster
        - 1 if the cluster is a mixed-match cluster
        - 2 if the cluster is an all-match cluster
        The prediction works in two steps:
        - predict the cluster for the row
        - for the cluster, returns 0,1 or 2 depending on the nomatch/mixedmatch/allmatch status of the cluster

        Args:
            X (np.ndarray/pd.DataFrame): score matrix

        Returns:
            np.ndarray: (0 for sure non matches, 1 for mixed matches, 2 for sure positive matches)
        """
        y_cluster = self.pred_cluster(X=X)
        y_pred = self._clusterclassifier.predict(X=y_cluster)
        return y_pred

    def transform(self, X):
        """
        Same as self.predict
        Args:
            X (pd.DataFrame/np.ndarray):

        Returns:
            np.ndarray: (0 for sure non matches, 1 for mixed matches, 2 for sure positive matches)
        """
        return self.predict(X=X)

    def fit_transform(self, X, y):
        self.fit(X=X, y=y, fit_cluster=True)
        return self.transform(X=X)

    def fit_predict(self, X, y):
        self.fit(X=X, y=y, fit_cluster=True)
        return self.predict(X=X)
