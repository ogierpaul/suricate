from sklearn.base import TransformerMixin
from suricate.preutils import createmultiindex
from suricate.dftransformers import DfVisualSbs, cartesian_join
import pandas as pd
from suricate.base import ConnectorMixin

class DfConnector(ConnectorMixin):
    """
    This connector (see the base class for connectors) will connect two dataframes, (one 'source' and one 'target').
    """
    def __init__(self, scorer, ixname='ix', source_suffix='source', target_suffix='target'):
        """

        Args:
            ixname:
            source_suffix:
            target_suffix:
            scorer (TransformerMixin): score pipeline or featureunion
        """
        ConnectorMixin.__init__(self, ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix)
        self.scorer = scorer

    def fit(self, X, y=None):
        self.scorer.fit(X=X, y=y)
        return self

    def transform(self, X):
        """

        Args:
            X (list): [df_source, df_target]

        Returns:
            pd.DataFrame: with index
        """
        Xt = self.scorer.transform(X=X)
        Xt = pd.DataFrame(data=Xt, index=self.getindex(X=X), columns=self.scorer.get_feature_names())
        return Xt

    def getindex(self, X):
        return createmultiindex(X=X, names=self.ixnamepairs)

    def getsbs(self, X, on_ix=None):
        if on_ix is None:
            on_ix = self.getindex(X=X)
        Xt = cartesian_join(source=X[0], target=X[0], on_ix=on_ix, ixname=self.ixname, source_suffix=self.source_suffix, target_suffix=self.target_suffix)
        return Xt

    def fetch_source(self, X, ix):
        return X[0].loc[ix]

    def fetch_target(self, X, ix):
        return X[1].loc[ix]
