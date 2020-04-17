import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class QuestionsMixin(TransformerMixin):
    """
    Base class for the Simple Questions and Hard Questions.
    For each cluster of the self.clusters, will generate at most self.n_questions
    """
    def __init__(self, n_questions=10):
        """

        Args:
            n_questions (int): number of lr_explore to be asked for each cluster
        """
        TransformerMixin.__init__(self)
        self.n_questions = n_questions
        self.n_clusters = None
        self.clusters = None

    def _transform(self, X):
        """
        Args:
            X (pd.Series): Vector of shape (n_pairs, 1) or (n_pairs,) with the cluster classifications of the pairs

        Returns:
            pd.MultiIndex: index number  of lines to take; dimension maximum is (n_clusters * n_questions, ) (some clusters may have a size inferior to n_questions)
        """
        if not isinstance(X, pd.Series):
            raise IndexError('Data must be a 1-dimensionnal pandas series with index')

        questions = pd.MultiIndex(levels=[[],[]],
                             codes=[[],[]],
                             names=X.index.names)
        for c in self.clusters:
            cluster = X.loc[X == c]
            if cluster.shape[0] < self.n_questions:
                sample_ix = cluster.index
            else:
                sample_ix = cluster.sample(self.n_questions).index
            questions = questions.union(sample_ix)
        return questions

    def transform(self, X):
        """
        Generate sample indexes.
        Args:
            X:

        Returns:
            pd.MultiIndex
        """
        return self._transform(X)

    def fit_predict(self,X, y=None):
        return self.fit_transform(X, y=y)

def cluster_stats(X, y_cluster, y_true):
    """

    Args:
        X (pd.DataFrame): similarity matrix (n_samples, n_features)
        y_cluster (pd.Series): y_cluster (n_samples, )
        y_true (pd.Series): y_true, labelled data (n_y_true, )
    Returns:
        pd.DataFrame: (n_clusters, 2):
    """
    y_avg_score = pd.Series(data=X.mean(axis=1), name='avg_score', index=X.index)
    X = pd.DataFrame(index=X.index)
    X['y_cluster'] = y_cluster
    X['avg_score'] = y_avg_score
    X_pivot = pd.pivot_table(data=X, index='y_cluster', values='avg_score', aggfunc=np.mean)
    X_matches = cluster_matches(y_cluster=y_cluster, y_true=y_true, normalize=False)
    y_pct_match = X_matches[1]/(X_matches[0]+ X_matches[1])
    X_pivot['pct_match'] = y_pct_match
    X_pivot.sort_values(by='pct_match', ascending=False, inplace=True)
    return X_pivot


def cluster_matches(y_cluster, y_true, normalize='index'):
    """

    Args:
        y_cluster (pd.Series): series with index
        y_true (pd.Series): series with index
        normalize:

    Returns:
        pd.DataFrame: cols [0,1], index = (n_clusters)
    """
    ix_common = y_cluster.index.intersection(y_true.index)
    df = pd.crosstab(index=y_cluster.loc[ix_common], columns=y_true.loc[ix_common], normalize=normalize)
    df.sort_values(by=1, ascending=False, inplace=True)
    assert isinstance(df, pd.DataFrame)
    return df