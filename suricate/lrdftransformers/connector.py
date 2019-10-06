from sklearn.base import TransformerMixin
from suricate.preutils import concatixnames, addsuffix, createmultiindex
from suricate.lrdftransformers import LrDfVisualHelper, create_lrdf_sbs
import pandas as pd


class LrDfConnector(TransformerMixin):
    def __init__(self, scorer, ixname='ix', lsuffix='left', rsuffix='right'):
        """

        Args:
            ixname:
            lsuffix:
            rsuffix:
            scorer (TransformerMixin): score pipeline or featureunion
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
        self.scorer = scorer
        #TODO: Add method to get scorenames

    def fit(self, X, y=None):
        self.scorer.fit(X=X, y=y)
        return self

    def transform(self, X):
        """

        Args:
            X (list): [df_left, df_right]

        Returns:
            np.ndarray
        """
        Xt = self.scorer.transform(X=X)
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

    def multiindex21column(self, on_ix, sep='-'):
        df = pd.DataFrame(index=on_ix)
        df.reset_index(inplace=True, drop=False)
        df[self.ixname] = df[self.ixnameleft] + sep + df[self.ixnameright]
        df.set_index(self.ixname, inplace=True, drop=True)
        return df.index
