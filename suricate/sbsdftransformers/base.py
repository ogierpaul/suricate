# -*- coding: utf-8 -*-
import pandas as pd
from dask import dataframe as dd
from sklearn.base import TransformerMixin


class BaseSbsComparator(TransformerMixin):
    def __init__(self, on_source='source', on_target='target', compfunc=None, n_jobs=1, *args, **kwargs):
        """
        base class for all transformers
        Args:
            on_source (str): name of suffix
            on_target (str):
            compfunc (callable): ['fuzzy', 'token', 'exact']
            n_jobs (int): number of parallel jobs. If n_jobs>1, will call the dask dataframe API
        """
        TransformerMixin.__init__(self)
        self.left = on_source
        self.right = on_target
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc
        self.n_jobs = n_jobs

    def transform(self, X):
        """
        Apply the compfunc to the on_source and on_target column
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        if self.n_jobs == 1:
            return self._stransform(X)
        else:
            return self._ptransform(X)

    def _stransform(self, X):
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

    def _ptransform(self, X):
        #TODO: Remove
        """
        Use of dask dataframe API to parall
        Args:
            X:

        Returns:
            np.array
        """
        y = dd.from_pandas(
            X.reset_index(drop=True),
            npartitions=self.n_jobs
        ).map_partitions(
            func=self.transform
        )
        return y

    def fit(self, *_):
        return self
