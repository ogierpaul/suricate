import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from suricate.preutils import concatixnames, addsuffix, createmultiindex

class DfTransformerMixin(TransformerMixin):
    def __init__(self, on=None, ixname='ix',
                 source_suffix='source', target_suffix='target', scoresuffix='score', **kwargs):
        """
        Args:
            ixname (str): name of the index, default 'ix'
            source_suffix (str): suffix to be added to the left dataframe default 'left', gives --> 'ix_source'
            target_suffix (str): suffix to be added to the left dataframe default 'right', gives --> 'ixright'
            on (str): name of the column on which to do the join
            scoresuffix (str): suffix to be attached to the on column name
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
            X (list): [df_source, df_target]
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
            X (list): [df_source, df_target]

        Returns:
            pd.Series: {['ix_source', 'ix_target']:score}
        """
        return pd.Series(
            data=self.transform(X=X),
            index=self._getindex(X=X),
            name=self.outcol)

    def show_pairs(self, X, y=None, use_cols=None):
        """
        Create a side by side table from a list of pairs (as a DataFrame)
        Args:
            X (list): of the form [df_source, df_target]
            y (pd.DataFrame/pd.Series): of the form {['ix_source', 'ix_target']:['y_true']}
            use_cols (list): columns to use

        Returns:
            pd.DataFrame {['ix_source', 'ix_target'] : ['name_source', 'name_target', .....]}
        """
        source = X[0]
        target = X[1]

        if y is None:
            xpairs = pd.DataFrame(index=self._getindex(X=X))
        elif isinstance(y, pd.DataFrame):
            xpairs = y.copy()
        else:
            assert isinstance(y, pd.Series)
            xpairs = pd.DataFrame(y.copy())

        xpairs = xpairs.reset_index(drop=False)

        if use_cols is None or len(use_cols) == 0:
            use_cols = source.columns.intersection(target.columns)
        xsource = source[use_cols].copy().reset_index(drop=False)
        xtarget = target[use_cols].copy().reset_index(drop=False)
        xsource = addsuffix(xsource, self.source_suffix).set_index(self.ixnamesource)
        xtarget = addsuffix(xtarget, self.target_suffix).set_index(self.ixnametarget)

        sbs = xpairs.join(
            xsource, on=self.ixnamesource, how='left'
        ).join(
            xtarget, on=self.ixnametarget, how='left'
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
        X (list): [df_source, df_target]
        Returns:
            np.ndarray: of shape(n_samples_source * n_samples_target, 1)
        """
        source = X[0]
        target = X[1]
        assert isinstance(source, pd.DataFrame)
        assert isinstance(target, pd.DataFrame)
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

    def _toseries(self, source, target, on_ix):
        """
        convert to series without nulls and copy
        Args:
            source (pd.Series/pd.DataFrame):
            target (pd.Series/pd.DataFrame):

        Returns:
            pd.Series, pd.Series
        """
        newsource = pd.Series()
        newtarget = pd.Series()
        if isinstance(source, pd.DataFrame):
            newsource = source[self.on].dropna().copy()
        if isinstance(target, pd.DataFrame):
            newtarget = target[self.on].dropna().copy()
        if isinstance(source, pd.Series):
            newsource = source.dropna().copy()
        if isinstance(target, pd.Series):
            newtarget = target.dropna().copy()
        for s, c in zip(['source', 'target'], [source, target]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        if on_ix is not None:
            newsource = newsource.loc[on_ix.levels[0].intersection(newsource.index)]
            newsource.index.name = self.ixname
            newtarget = newtarget.loc[on_ix.levels[1].intersection(newtarget.index)]
            newtarget.index.name = self.ixname
        return newsource, newtarget

    def _todf(self, source, target, on_ix=None):
        """
        convert to dataframe with one column withoutnulls and copy
        Args:
            source (pd.Series/pd.DataFrame):
            target (pd.Series/pd.DataFrame):

        Returns:
            pd.DataFrame, pd.DataFrame
        """
        newsource = pd.DataFrame()
        newtarget = pd.DataFrame()
        if isinstance(source, pd.DataFrame):
            if self.on is not None and self.on != 'none':
                newsource = source[[self.on]].dropna(subset=[self.on]).copy()
            else:
                newsource = source.copy()
        if isinstance(target, pd.DataFrame):
            if self.on is not None and self.on != 'none':
                newtarget = target[[self.on]].dropna(subset=[self.on]).copy()
            else:
                newtarget = target.copy()

        if isinstance(source, pd.Series):
            newsource = pd.DataFrame(source.dropna().copy())
        if isinstance(target, pd.Series):
            newtarget = pd.DataFrame(target.dropna().copy())
        for s, c in zip(['source', 'target'], [source, target]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        if on_ix is None:
            return newsource, newtarget
        else:
            newsource = newsource.loc[on_ix.levels[0].intersection(newsource.index)]
            newsource.index.name = self.ixname
            newtarget = newtarget.loc[on_ix.levels[1].intersection(newtarget.index)]
            newtarget.index.name = self.ixname
            return newsource, newtarget

class DfIndexEncoder(TransformerMixin):
    def __init__(self, ixname='ix', source_suffix='source', target_suffix='target'):
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix
        )
        self.index = pd.Index
        self.dfnum = pd.DataFrame()
        self.dfix = pd.DataFrame()
        self.num = None

    def fit(self, X, y=None):
        """

        Args:
            X (list): [df_source, df_target]
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



