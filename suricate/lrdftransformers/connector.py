from sklearn.base import TransformerMixin
from suricate.preutils import concatixnames, addsuffix, createmultiindex
from suricate.lrdftransformers import LrDfVisualHelper, create_lrdf_sbs
import pandas as pd
from suricate.base import ConnectorMixin


class LrDfConnector(ConnectorMixin):
    def __init__(self, scorer, ixname='ix', lsuffix='left', rsuffix='right'):
        """

        Args:
            ixname:
            lsuffix:
            rsuffix:
            scorer (TransformerMixin): score pipeline or featureunion
        """
        ConnectorMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        self.scorer = scorer

    def fit(self, X, y=None):
        self.scorer.fit(X=X, y=y)
        return self

    def transform(self, X):
        """

        Args:
            X (list): [df_left, df_right]

        Returns:
            pd.DataFrame: with index
        """
        Xt = self.scorer.transform(X=X)
        Xt = pd.DataFrame(data=Xt, index=self.getindex(X=X))
        return Xt

    def getindex(self, X):
        return createmultiindex(X=X, names=self.ixnamepairs)

    def getsbs(self, X, on_ix=None):
        if on_ix is None:
            on_ix = self.getindex(X=X)
        Xt = create_lrdf_sbs(X=X, on_ix=on_ix, ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix)
        return Xt

    def fetch_left(self, X, ix):
        return X[0].loc[ix]

    def fetch_right(self, X, ix):
        return X[1].loc[ix]
