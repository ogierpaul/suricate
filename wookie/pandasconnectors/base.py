import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union

from wookie.obsolete import evalprecisionrecall
from wookie.preutils import concatixnames, addsuffix


# TODO: remove the pruning_ths / use the y in .transform
# Not to self: do not see the problem yet / What is this function?

def cartesian_join(left, right, lsuffix='left', rsuffix='right'):
    """

    Args:
        left (pd.DataFrame): table 1
        right (pd.DataFrame): table 2
        lsuffix (str):
        rsuffix (str):

    Returns:
        pd.DataFrame

    Examples:
        df1 = pd.DataFrame({'a':['foo', 'bar']})
             a
        0   foo
        1	bar

        df2 = pd.DataFrame({'b':['foz', 'baz']})
             b
        0   foz
        1	baz

        cartesian_join(df1, df2)
            index_left	a_left	index_right	b_right
        0	0	        foo	    0	        foz
        1	0	        foo	    1	        baz
        2	1	        bar	    0	        foz
        3	1	        bar	    1	        baz

    """

    def rename_with_suffix(df, suffix):
        """
        rename the columns with a suffix, including the index
        Args:
            df (pd.DataFrame): {'ix':['name']}
            suffix (str): 'left'

        Returns:
            pd.DataFrame

        Examples:
            df = pd.DataFrame({'a':['foo', 'bar']})
                 a
            0   foo
            1	bar

            rename_with_suffix(df, 'right')

                index_right	a_right
            0	0	        foo
            1	1	        bar
        """
        if suffix is None:
            return df
        assert isinstance(suffix, str)
        assert isinstance(df, pd.DataFrame)
        df_new = df.copy()
        if df.index.name is None:
            ixname = 'ix'
        else:
            ixname = df.index.name
        df_new.index.name = ixname
        df_new.reset_index(drop=False, inplace=True)
        cols = df_new.columns
        mydict = dict(
            zip(
                cols,
                map(lambda c: c + '_' + suffix, cols)
            )
        )
        df_new.rename(columns=mydict, inplace=True)
        return df_new

    # hack to create a column name unknown to both df1 and df2
    tempcolname = 'f1b3'
    while tempcolname in left.columns or tempcolname in right.columns:
        tempcolname += 'f'

    # create a new df1 with renamed cols
    df1new = rename_with_suffix(left, lsuffix)
    df2new = rename_with_suffix(right, rsuffix)
    df1new[tempcolname] = 0
    df2new[tempcolname] = 0
    dfnew = pd.merge(df1new, df2new, on=tempcolname).drop([tempcolname], axis=1)
    del df1new, df2new, tempcolname

    return dfnew


class DFConnector(TransformerMixin):
    def __init__(self, on=None, ixname='ix',
                 lsuffix='left', rsuffix='right', scoresuffix='score',
                 n_jobs=1, pruning_ths=None, **kwargs):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            on (str): name of the column on which to do the join
            scoresuffix (str): name of the score suffix
            n_jobs (int):
            pruning_ths (float): return only the pairs which have a score greater than the store_ths.
        """
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
        if self.on == None:
            self.outcol = self.scoresuffix
        else:
            self.outcol = self.on + '_' + self.scoresuffix
        self.n_jobs = n_jobs
        self.pruning_ths = pruning_ths
        self.fitted = False
        pass

    def _getindex(self, X, y=None):
        """
        Return the cartesian product index of both dataframes
        Args:
            X:
            y (pd.Series/pd.DataFrame/pd.MultiIndex): dummy, not used

        Returns:
            pd.MultiIndex
        """
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

    def showpairs(self, X, y=None, use_cols=None):
        """
        Create a side by side table from a list of pairs (as a DataFrame)
        Args:
            y (pd.DataFrame/pd.Series): of the form {['ix_left', 'ix_right']:['y_true']}
            use_cols (list): columns to use

        Returns:
            pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
        """
        left = X[0]
        right = X[1]

        if y is None:
            xpairs = pd.DataFrame(index=self._getindex(X=X, y=y))
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

    def transform(self, X, y=None, as_series=False):
        """
        X (list): [left_df, right_df]
        y (pd.Series) : dummy value
        Returns:
            np.ndarray
        """
        left = X[0]
        right = X[1]
        assert isinstance(left, pd.DataFrame)
        assert isinstance(right, pd.DataFrame)
        ix = self._getindex(X=X, y=y)
        score = self._transform(X=X, on_ix=ix)
        test_remove_pruningths = False
        # TODO: REMOVE FEATURE FLIPPING
        if test_remove_pruningths:
            # This code here leads to bug
            commonindex = score.index.intersection(ix)
            # y_pred.loc[commonindex] = score[commonindex]
            return score
        else:
            if self.pruning_ths is None:
                y_pred = pd.Series(
                    index=ix,
                    name=self.outcol
                )
                commonindex = score.index.intersection(y_pred.index)
                y_pred.loc[commonindex] = score[commonindex]
            else:
                y_pred = score.loc[score >= self.pruning_ths]
            if as_series is False:
                y_pred = y_pred.values.reshape(-1, 1)
            return y_pred

    def _transform(self, X, on_ix=None):
        """

        Args:
            X:
            on_ix:

        Returns:
            pd.Series()
        """
        return pd.Series(index=on_ix)

    def pruning_score(self, X, y_true):
        """
        compression: defined by the number of possible pairs divided by the number of actual pairs
        precision and recall : depends on y_true
        Args:
            y_true: list of pairs in the index

        Returns:
            dict: ['compression', 'precision', 'recall']
        """
        score = dict()
        y_pred = self.transform(X=X, y=y_true, as_series=True)
        score['compression'] = (X[0].shape[0] * X[1].shape[0]) / y_pred.shape[0]
        precision, recall = evalprecisionrecall(y_true=y_true, y_pred=y_pred)
        score['precision'] = precision
        score['recall'] = recall
        return score

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


class DfFeatureUnion(DFConnector):
    def __init__(self, stages, *args, **kwargs):
        DFConnector.__init__(self, args, kwargs)
        self.stages = stages

    def _fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None, as_series=False):
        if as_series is False:
            pipe = make_union(*self.stages)
            return pipe.transform(X=X)
        else:
            df = pd.DataFrame(index=self._getindex(X=X, y=y))
            for con in self.stages:
                df[con.outcol] = con.transform(X=X, y=y, as_series=True)
            return df
