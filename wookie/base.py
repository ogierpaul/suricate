# -*- coding: utf-8 -*-
import pandas as pd
from dask import dataframe as dd
from sklearn.base import TransformerMixin

from wookie.connectors.dataframes.base import evalprecisionrecall
from wookie.preutils import concatixnames


class BaseLrComparator(TransformerMixin):
    """
    This is the base Left-Right Comparator
    Idea is that is should have take a left dataframe, a right dataframe,
    and return a combination of two, with a comparison score
    """

    def __init__(self,
                 on=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 scoresuffix='score',
                 store_threshold=0.0,
                 n_jobs=2
                 ):
        """

        Args:
            on(str): column to use on the left and right df
            ixname (str): name of the index of left and right
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): score suffix: the outputvector has the name on + '_' + scoresuffix
            store_threshold (float): threshold to use to store the relevance score
            n_jobs (float): number of parallel jobs
        """
        TransformerMixin.__init__(self)
        if on is None:
            on = 'none'
        self.on = on
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.scoresuffix = scoresuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix
        )
        self.store_threshold = store_threshold
        self.outcol = '_'.join([self.on, self.scoresuffix])
        self.n_jobs = n_jobs
        pass

    def _toseries(self, left, right):
        """
        convert to series withoutnulls and copy
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):

        Returns:
            pd.Series, pd.Series
        """
        newleft = pd.Series()
        newright = pd.Series()
        if isinstance(left, pd.DataFrame):
            newleft = left[self.on].dropna().copy()
        if isinstance(right, pd.DataFrame):
            newright = right[self.on].dropna().copy()
        if isinstance(left, pd.Series):
            newleft = left.dropna().copy()
        if isinstance(right, pd.Series):
            newright = right.dropna().copy()
        for s, c in zip(['left', 'right'], [left, right]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        return newleft, newright

    def _todf(self, left, right):
        """
        convert to dataframe with one column withoutnulls and copy
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):

        Returns:
            pd.DataFrame, pd.DataFrame
        """
        newleft = pd.DataFrame()
        newright = pd.DataFrame()
        if isinstance(left, pd.DataFrame):
            if self.on is not None and self.on != 'none':
                newleft = left[[self.on]].dropna(subset=[self.on]).copy()
            else:
                newleft = left.copy()
        if isinstance(right, pd.DataFrame):
            if self.on is not None and self.on != 'none':
                newright = right[[self.on]].dropna(subset=[self.on]).copy()
            else:
                newright = right.copy()

        if isinstance(left, pd.Series):
            newleft = pd.DataFrame(left.dropna().copy())
        if isinstance(right, pd.Series):
            newright = pd.DataFrame(right.dropna().copy())
        for s, c in zip(['left', 'right'], [left, right]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        return newleft, newright

    def evalscore(self, left, right, y_true):
        """
        evaluate precision and recall
        Args:
            left (pd.DataFrame/pd.Series):
            right (pd.DataFrame/pd.Series):
            y_true (pd.Series):

        Returns:
            float, float: precision and recall
        """
        # assert hasattr(self, 'transform') and callable(getattr(self, 'transform'))
        # noinspection
        y_pred = self.transform(left=left, right=right)
        precision, recall = evalprecisionrecall(y_true=y_true, y_pred=y_pred)
        return precision, recall


class BaseSbsComparator(TransformerMixin):
    def __init__(self, on_left='left', on_right='right', compfunc=None, n_jobs=1, *args, **kwargs):
        """
        base class for all transformers
        Args:
            on_left (str):
            on_right (str):
            compfunc (callable): ['fuzzy', 'token', 'exact']
            n_jobs (int): number of parallel jobs. If n_jobs>1, will call the dask dataframe API
        """
        TransformerMixin.__init__(self)
        self.left = on_left
        self.right = on_right
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc
        self.n_jobs = n_jobs

    def transform(self, X):
        """
        Apply the compfunc to the on_left and on_right column
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
        """
        Use of dask dataframe API to parall
        Args:
            X:

        Returns:

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
