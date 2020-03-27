import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from suricate.preutils import concatixnames, addsuffix, createmultiindex

class LrDfTransformerMixin(TransformerMixin):
    def __init__(self, on=None, ixname='ix',
                 lsuffix='left', rsuffix='right', scoresuffix='score', **kwargs):
        """
        Args:
            ixname (str): name of the index, default 'ix'
            lsuffix (str): suffix to be added to the left dataframe default 'left', gives --> 'ix_left'
            rsuffix (str): suffix to be added to the left dataframe default 'right', gives --> 'ixright'
            on (str): name of the column on which to do the join
            scoresuffix (str): suffix to be attached to the on column name
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
        self.on = on
        self.scoresuffix = scoresuffix
        if self.on is None:
            self.outcol = self.scoresuffix
        else:
            self.outcol = self.on + '_' + self.scoresuffix
        self.fitted = False
        pass

    def _getindex(self, X, y=None):
        """
        Return the cartesian product index of both dataframes
        Args:
            X (list): [df_left, df_right]
            y (pd.Series/pd.DataFrame/pd.MultiIndex): dummy, not used

        Returns:
            pd.MultiIndex
        """
        ix = createmultiindex(X=X, names=self.ixnamepairs)
        return ix

    def transformtoseries(self, X):
        """
        Transform and add index and name to transform it into a series
        Args:
            X (list): [df_left, df_right]

        Returns:
            pd.Series: {['ix_left', 'ix_right']:score}
        """
        return pd.Series(
            data=self.transform(X=X),
            index=self._getindex(X=X),
            name=self.outcol)

    def show_pairs(self, X, y=None, use_cols=None):
        """
        Create a side by side table from a list of pairs (as a DataFrame)
        Args:
            X (list): of the form [df_left, df_right]
            y (pd.DataFrame/pd.Series): of the form {['ix_left', 'ix_right']:['y_true']}
            use_cols (list): columns to use

        Returns:
            pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
        """
        left = X[0]
        right = X[1]

        if y is None:
            xpairs = pd.DataFrame(index=self._getindex(X=X))
        elif isinstance(y, pd.DataFrame):
            xpairs = y.copy()
        else:
            assert isinstance(y, pd.Series)
            xpairs = pd.DataFrame(y.copy())

        xpairs = xpairs.reset_index(drop=False)

        if use_cols is None or len(use_cols) == 0:
            use_cols = left.columns.intersection(right.columns)
        xleft = left[use_cols].copy().reset_index(drop=False)
        xright = right[use_cols].copy().reset_index(drop=False)
        xleft = addsuffix(xleft, self.lsuffix).set_index(self.ixnameleft)
        xright = addsuffix(xright, self.rsuffix).set_index(self.ixnameright)

        sbs = xpairs.join(
            xleft, on=self.ixnameleft, how='left'
        ).join(
            xright, on=self.ixnameright, how='left'
        ).set_index(
            self.ixnamepairs
        )
        return sbs

    def fit(self, X=None, y=None):
        self.fitted = True
        return self._fit(X=X, y=None)

    def _fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """
        X (list): [left_df, right_df]
        Returns:
            np.ndarray: of shape(n_samples_left * n_samples_right, 1)
        """
        left = X[0]
        right = X[1]
        assert isinstance(left, pd.DataFrame)
        assert isinstance(right, pd.DataFrame)
        ix = self._getindex(X=X)
        score = self._transform(X=X)
        return score

    def _transform(self, X):
        """

        Args:
            X:

        Returns:
            numpy.ndarray
        """
        return np.zeros(shape=(X[0].shape[0] * X[1].shape[0], 1))

    def _toseries(self, left, right, on_ix):
        """
        convert to series without nulls and copy
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
        if on_ix is not None:
            newleft = newleft.loc[on_ix.levels[0].intersection(newleft.index)]
            newleft.index.name = self.ixname
            newright = newright.loc[on_ix.levels[1].intersection(newright.index)]
            newright.index.name = self.ixname
        return newleft, newright

    def _todf(self, left, right, on_ix=None):
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
        if on_ix is None:
            return newleft, newright
        else:
            newleft = newleft.loc[on_ix.levels[0].intersection(newleft.index)]
            newleft.index.name = self.ixname
            newright = newright.loc[on_ix.levels[1].intersection(newright.index)]
            newright.index.name = self.ixname
            return newleft, newright

class LrDfIndexEncoder(TransformerMixin):
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right'):
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        self.index = pd.Index
        self.dfnum = pd.DataFrame()
        self.dfix = pd.DataFrame()
        self.num = None

    def fit(self, X, y=None):
        """

        Args:
            X (list): [left_df, right_df]
            y: dummy, not used

        Returns:
            self
        """
        self.index = createmultiindex(X=X, names=self.ixnamepairs)
        self.dfnum = pd.Series(index=np.arange(0, len(self.index)), data=self.index.values, name='ix')
        self.dfix = pd.Series(index=self.index, data=np.arange(0, len(self.index)), name='ixnum')
        return self

    def transform(self, X, y=None):
        return self.index

    def ix_to_num(self, ix):
        """

        Args:
            ix (np.ndarray/pd.Index): double index

        Returns:
            np.ndarray: monotonic index values
        """
        return self.dfix.loc[ix]

    def num_to_ix(self, vals):
        """

        Args:
            vals (np.ndarray): list of values (monotonic index)

        Returns:
            np.ndarray
        """
        return self.dfnum.loc[vals]



