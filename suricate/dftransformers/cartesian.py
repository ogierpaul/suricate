import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion

from suricate.dftransformers import DfTransformerMixin
from suricate.preutils import concatixnames, createmultiindex
import itertools


class DfVisualSbs(TransformerMixin):
    def __init__(self, ixname='ix', source_suffix='source', target_suffix='target', usecols=None, **kwargs):
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix
        )
        self.usecols = usecols
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

    def fit(self, X=None, y=None):
        """
        Dummy, do nothing
        Args:
            X:
            y:

        Returns:

        """
        return self

    def transform(self, X=None, y=None):
        """

        Args:
            X (list): [df_source, df_target]
            y: dummy

        Returns:
            pd.DataFrame: with index [ix_source, ix_target]
        """
        X_sbs = cartesian_join(source=X[0], target=X[1], source_suffix=self.source_suffix, target_suffix=self.target_suffix)
        return X_sbs


def cartesian_join(source, target, source_suffix='source', target_suffix='target', ixname='ix', on_ix=None, usecols=None):
    """

    Args:
        source (pd.DataFrame): table 1
        target (pd.DataFrame): table 2
        source_suffix (str): suffix, example 'source'
        target_suffix (str): example 'target'
        ixname (str): default 'ix'
        on_ix (MultiIndex): only take this particular index, optional
        usecols (list): list of columns to display. Optional. If None take the intersection

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
            index_source	a_source	index_target	b_target
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
            suffix (str): example 'source'

        Returns:
            pd.DataFrame

        Examples:
            df = pd.DataFrame({'a':['foo', 'bar']})
                 a
            0   foo
            1	bar

            rename_with_suffix(df, 'target')

                index_target	a_target
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
    # Find the common columns:
    if usecols is None:
        usecols = source.columns.intersection(target.columns)

    if on_ix is None:
        # hack to create a column name unknown to both df1 and df2
        tempcolname = 'f1b3'
        while tempcolname in source.columns or tempcolname in target.columns:
            tempcolname += 'f'

        # create a new df1 with renamed cols
        df1new = rename_with_suffix(source[usecols], source_suffix)
        df2new = rename_with_suffix(target[usecols], target_suffix)
        df1new[tempcolname] = 0
        df2new[tempcolname] = 0

        # Use pd.merge to do the join
        dfnew = pd.merge(df1new, df2new, on=tempcolname).drop([tempcolname], axis=1)
        ixname = source.index.name
        ixnamesource = ixname + '_' + source_suffix
        ixnametarget = ixname + '_' + target_suffix
        dfnew.set_index(
            [ixnamesource, ixnametarget],
            drop=True,
            inplace=True
        )
        del df1new, df2new, tempcolname
    else:
        # if on_ix is not False, do two separate left joins
        ixname = source.index.name
        ixnamesource = ixname + '_' + source_suffix
        ixnametarget = ixname + '_' + target_suffix
        dfnew = pd.DataFrame(
            index=on_ix
        ).reset_index(
            drop=False
        ).join(
            rename_with_suffix(source[usecols], source_suffix).set_index(ixnamesource),
            on=ixnamesource,
            how='left'
        ).join(
            rename_with_suffix(target[usecols], target_suffix).set_index(ixnametarget),
            on=ixnametarget,
            how='left'
        ).set_index(
            [ixnamesource, ixnametarget],
            drop=True
        )
    # Reorder the columns and set the index
    order_cols = [('{}_{}'.format(c, source_suffix), '{}_{}'.format(c, target_suffix)) for c in usecols]
    order_cols = list(itertools.chain(*order_cols))
    dfnew = dfnew.loc[:, order_cols]
    return dfnew

