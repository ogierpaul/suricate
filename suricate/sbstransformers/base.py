# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import TransformerMixin


class BaseSbsComparator(TransformerMixin):
    def __init__(self, on_source='source', on_target='target', compfunc=None,  *args, **kwargs):
        """
        base class for all transformers
        Args:
            on_source (str): name of suffix
            on_target (str):
            compfunc (callable): ['fuzzy', 'token', 'exact']
        """
        TransformerMixin.__init__(self)
        self.left = on_source
        self.right = on_target
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc

    def transform(self, X):
        """
        Apply the compfunc to the on_source and on_target column
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        return self._transform(X)

    def _transform(self, X):
        """
        Pandas apply function on rows (axis=1).
        Args:
            X (pd.DataFrame):

        Returns:
            np.array
        """
        y = X.apply(
            lambda r: self.compfunc(
                r.loc[self.left],
                r.loc[self.right]
            ),
            axis=1
        ).values.reshape(-1, 1)
        return y

    def fit(self, *_):
        return self
