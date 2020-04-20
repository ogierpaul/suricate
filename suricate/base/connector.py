from sklearn.base import TransformerMixin
from suricate.preutils import concatixnames
import pandas as pd


class ConnectorMixin(TransformerMixin):
    def __init__(self, ixname='ix', source_suffix='source', target_suffix='target'):
        """

        Args:
            ixname: 'ix'
            source_suffix: 'source'
            target_suffix: 'target'
        """
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix
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

    def getsbs(self, X, on_ix=None):
        """

        Args:
            X: input data
            on_ix (pd.MultiIndex): Optional, specify the index on which you want the side-by-side view

        Returns:
            pd.DataFrame
        """
        Xt = pd.DataFrame()
        return Xt

    def fetch_source(self, X, ix):
        """

        Args:
            X:
            ix (pd.Index):

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame()

    def fetch_target(self, X, ix):
        """

        Args:
            X:
            ix (pd.Index):

        Returns:
            pd.DataFrame
        """
        return pd.DataFrame()

    def fit_transform(self, X, y=None, **fit_params):
        """
        Will send back the similarity matrix of the connector with the index as DataFrame
        Args:
            X: input data
            y:
            **fit_params:

        Returns:
            pd.DataFrame
        """
        self.fit(X=X, y=y, **fit_params)
        return self.transform(X=X)

    def multiindex21column(self, on_ix, sep='-'):
        """

        Args:
            on_ix (pd.MultiIndex): two level multi index (ix_source, ix_target)
            sep: separator

        Returns:
            pd.Index: name 'ix', concatenation of ix_source, sep, on ix_target
        """
        df = pd.DataFrame(index=on_ix)
        df.reset_index(inplace=True, drop=False)
        df[self.ixname] = df[[self.ixnamesource, self.ixnametarget]].astype(str).agg(str(sep).join, axis=1)
        df.set_index(self.ixname, inplace=True, drop=True)
        return df.index

def multiindex21column(on_ix, sep='-', ixname='ix', ixnamesource='ix_source', ixnametarget='ix_target'):
    """

    Args:
        on_ix (pd.MultiIndex): two level multi index (ix_source, ix_target)
        sep: separator

    Returns:
        pd.Index: name 'ix', concatenation of ix_source, sep, ix_target
    """
    df = pd.DataFrame(index=on_ix)
    df.reset_index(inplace=True, drop=False)
    df[ixname] = df[[ixnamesource, ixnametarget]].astype(str).agg(str(sep).join, axis=1)
    df.set_index(ixname, inplace=True, drop=True)
    return df.index
