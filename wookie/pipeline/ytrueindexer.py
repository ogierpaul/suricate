import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError


class REindedexer(TransformerMixin):
    def __init__(self, X, ixnamepairs, *args, **kwargs):
        TransformerMixin.__init__(self)
        self.on_ix = None
        self.fitted = False
        self.all_ix = pd.MultiIndex.from_product([X[0].index, X[1].index], names=ixnamepairs)
        self.ixnamepairs = ixnamepairs
        pass

    def fit(self, X=None, y=None):
        self.on_ix = self._getindex(X=X, y=y)
        pass

    def transform(self, X, y=None, as_series=False):
        if self.fitted is False:
            raise NotFittedError('Transformer is not fitted')
        else:
            if self.on_ix is None:
                return X
            else:
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
