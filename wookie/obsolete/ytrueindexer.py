import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError

from wookie.preutils import concatixnames


# THIS MODULE FAILS
# THIS LAST TEST FAILED AND I NEED TO REBUILD A NEW METHOD
# I HAVE A K.O because the output of a pipeline is a np.ndarray
# but my indexer has an index
# I cannot a posteriori link the nump.ndarray (X_score) with the y_true index


class NpIndexTransformer(TransformerMixin):
    """
    This class has to be able to do:
    - with X an array of n * m features
    - with y the position of arrays to keep
    - in fit (X, y), select the index to be y
    - in transform(X), return X indexed by y
    """

    def __init__(self, ixname='ix',
                 lsuffix='left',
                 rsuffix='right', *args, **kwargs):
        TransformerMixin.__init__(self)
        self.on_ix = None
        self.fitted = False
        pass

    def fit(self, X=None, y=None):
        """
        if y is None:
            self.on_ix = None --> return X
        else:
            if y is np.ndarray
                self.on_ix = y

        add self.fitted = True
        Args:
            X: default None, X= np.array of n * m
            y: default None

        Returns:
            self
        """
        self.on_ix = self._getindex(X=X, y=y)
        self.fitted = True
        return self

    def transform(self, X, y=None):
        if self.fitted is False:
            raise NotFittedError('Transformer is not fitted')
        else:
            if self.on_ix is None:
                return X
            else:
                assert isinstance(X, np.ndarray)
                return X[self.on_ix]

    def _getindex(self, X, y):
        return y


class LrIndexTransformer(TransformerMixin):
    """
    This class has to be able to do:
    - with X a [left, right] couple of dataframe
    - in fit (X, y), select the index to be y
    - in transform(X), return X indexed by y
    """

    def __init__(self, ixname='ix',
                 lsuffix='left',
                 rsuffix='right', *args, **kwargs):
        TransformerMixin.__init__(self)
        self.on_ix = None
        self.fitted = False
        self.all_ix = None
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        pass

    def fit(self, X=None, y=None):
        """
        if y is None:
            self.on_ix = MultiIndex (X[0], X[1]) --> cartesian product of left and right dataframe
        else:
            if y is Index
                self.on_ix = y
            else
                if y is Series or DataFrame --> return y.index
        add self.fitted = True
        Args:
            X: default None, X=(df_left, df_right)
            y: default None

        Returns:
            self
        """
        self.on_ix = self._getindex(X=X, y=y)
        self.fitted = True
        return self

    def transform(self, X, y=None, as_series=False):
        if self.fitted is False:
            raise NotFittedError('Transformer is not fitted')
        else:
            if self.on_ix is None:
                return X
            else:
                '''
                TODO: Correct
                AttributeError: 'numpy.ndarray' object has no attribute 'index'
                '''
                self.all_ix = pd.MultiIndex.from_product([X[0].index, X[1].index], names=ixnamepairs)
                X2 = pd.DataFrame(X, index=self.all_ix)
                X2 = X2.loc[self.on_ix]
                return X2

    def _getindex(self, X, y):
        if isinstance(y, pd.MultiIndex):
            return y
        elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            return y.index
        elif y is None:
            ix = pd.MultiIndex.from_product(
                [X[0].index, X[1].index],
                names=self.ixnamepairs
            )
            return ix
        else:
            print('index, series or dataframe or None expected')
            return y
