from sklearn.base import TransformerMixin
from suricate.preutils import concatixnames, addsuffix, createmultiindex
from suricate.lrdftransformers import LrDfVisualHelper, create_lrdf_sbs
import pandas as pd


class ConnectorMixin(TransformerMixin):
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right'):
        """

        Args:
            ixname: 'ix'
            lsuffix: 'left'
            rsuffix: 'right'
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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """

        Args:
            X:

        Returns:
            pd.DataFrame: with index
        """
        Xt = pd.DataFrame()
        return Xt

    def getindex(self, X):
        """

        Args:
            X:

        Returns:
            pd.MultiIndex
        """
        return pd.MultiIndex(names=self.ixnamepairs)

    def getsbs(self, X, on_ix=None):
        """

        Args:
            X:
            on_ix (pd.MultiIndex):

        Returns:
            pd.DataFrame
        """
        Xt = pd.DataFrame()
        return Xt

    def fetch_left(self, X, ix):
        """

        Args:
            X:
            ix (pd.Index):

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame()

    def fetch_right(self, X, ix):
        """

        Args:
            X:
            ix (pd.Index):

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame()

    def multiindex21column(self, on_ix, sep='-'):
        """

        Args:
            on_ix (pd.MultiIndex): two level multi index (ix_left, ix_right)
            sep: separator

        Returns:
            pd.Index: name 'ix', concatenation of ix_left, sep, on ix_right
        """
        df = pd.DataFrame(index=on_ix)
        df.reset_index(inplace=True, drop=False)
        df[self.ixname] = df[self.ixnameleft] + sep + df[self.ixnameright]
        df.set_index(self.ixname, inplace=True, drop=True)
        return df.index