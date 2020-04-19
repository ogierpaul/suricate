import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion

from suricate.dftransformers import DfTransformerMixin
from suricate.preutils import concatixnames, createmultiindex
import itertools

#TODO: Check Documentation and relevance of each of those cartesian operations

class CartesianSt(DfTransformerMixin):
    """
    This transformer returns the cartesian product of source and target indexes
    """

    def __init__(self, ixname='ix', source_suffix='source', target_suffix='target', on='all',
                 scoresuffix='cartesianscore', **kwargs):
        DfTransformerMixin.__init__(self, ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix, on=on,
                                    scoresuffix=scoresuffix, **kwargs)

    def _transform(self, X):
        """

        Args:
            X (list):

        Returns:
            np.ndarray: transformer returns the cartesian product of source and target indexes \
                of shape(n_samples_source * n_samples_target, 1)
        """
        return np.ones(shape=(X[0].shape[0] * X[1].shape[0], 1))


class CartesianDataPasser(TransformerMixin):
    """
    THIS CLASS IS NOT A DF CONNECTOR BUT A TRANSFORMER MIXIN
    It returns the cartesian join of the two dataframes with all their columns
    """

    def __init__(self, ixname='ix', source_suffix='source', target_suffix='target', **kwargs):
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix
        )

    def fit(self, X=None):
        return self

    def transform(self, X, y=None):
        return self._transform(X=X, y=None)

    def _fit(self, X=None, y=None):
        return self

    def _transform(self, X, y=None):
        return cartesian_join(source=X[0], target=X[1], source_suffix=self.source_suffix, target_suffix=self.target_suffix)

class DfVisualHelper(TransformerMixin):
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
        # Re-arrange the columns to put the same columns side-by-side
        mycols = [self.ixnamesource, self.ixnametarget]
        if self.usecols is None:
            usecols = X[0].columns
        else:
            usecols = self.usecols
        for c in usecols:
            mycols.append(c + '_' + self.source_suffix)
            mycols.append(c + '_' + self.target_suffix)
        X_sbs = X_sbs[mycols].set_index(self.ixnamepairs)
        return X_sbs


def create_sbs(X, on_ix=None, ixname ='ix', source_suffix='source', target_suffix='target'):
    """

    Args:
        X (list): [df_source, df_target]
        on_ix (pd.MultiIndex): collection of pairs ('ix_source', 'ix_target')

    Returns:
        pd.DataFrame, of shape (len(on_ix), df.shape[1] * 2 [{('ix_source', 'ix_target'):('name_source', 'name_target', ...}]
    """
    usecols = X[0].columns.intersection(X[1].columns)
    # Check on_ix is contained into the cartesian join of df_source and df_target
    allix = createmultiindex(X=X, names={'{}_{}'.format(ixname, source_suffix), '{}_{}'.format(ixname, target_suffix)})
    if len(on_ix.difference(allix)) > 0:
        raise IndexError(
            'Indexes called {} not found in cartesian product of source and target dataframe'.format(
                on_ix.difference(allix)
            )
        )

    # Rename the source and target dataframe with suffix
    df_source = X[0][usecols]
    df_source.columns = ['{}_{}'.format(c, source_suffix) for c in usecols]
    df_target = X[1][usecols]
    df_target.columns = [c + '_' + target_suffix for c in usecols]
    # Join on the index
    Xsbs = pd.DataFrame(index=on_ix).reset_index(drop=False)
    Xsbs = Xsbs.join(df_source, on='{}_{}'.format(ixname, source_suffix), how='left')
    Xsbs = Xsbs.join(df_target, on='{}_{}'.format(ixname, target_suffix), how='left')
    # Re-order the columns
    order_cols = [('{}_{}'.format(c, source_suffix), '{}_{}'.format(c, target_suffix)) for c in usecols]
    order_cols = list(itertools.chain(*order_cols))
    Xsbs = Xsbs.set_index(
        ['{}_{}'.format(ixname, source_suffix), '{}_{}'.format(ixname, target_suffix)],
        drop=True
    )
    Xsbs = Xsbs.loc[:, order_cols]
    return Xsbs


def cartesian_join(source, target, source_suffix='source', target_suffix='target', on_ix=None):
    """

    Args:
        source (pd.DataFrame): table 1
        target (pd.DataFrame): table 2
        source_suffix (str):
        target_suffix (str):
        on_ix (MultiIndex):

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
            suffix (str): 'left'

        Returns:
            pd.DataFrame

        Examples:
            df = pd.DataFrame({'a':['foo', 'bar']})
                 a
            0   foo
            1	bar

            rename_with_suffix(df, 'right')

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

    if on_ix is None:
        # hack to create a column name unknown to both df1 and df2
        tempcolname = 'f1b3'
        while tempcolname in source.columns or tempcolname in target.columns:
            tempcolname += 'f'

        # create a new df1 with renamed cols
        df1new = rename_with_suffix(source, source_suffix)
        df2new = rename_with_suffix(target, target_suffix)
        df1new[tempcolname] = 0
        df2new[tempcolname] = 0
        dfnew = pd.merge(df1new, df2new, on=tempcolname).drop([tempcolname], axis=1)
        del df1new, df2new, tempcolname
    else:
        ixname = source.index.name
        ixnamesource = ixname + '_' + source_suffix
        ixnametarget = ixname + '_' + target_suffix
        dfnew = pd.DataFrame(
            index=on_ix
        ).reset_index(
            drop=False
        ).join(
            rename_with_suffix(source, source_suffix).set_index(ixnamesource),
            on=ixnamesource,
            how='left'
        ).join(
            rename_with_suffix(target, target_suffix).set_index(ixnametarget),
            on=ixnametarget,
            how='left'
        ).set_index(
            [ixnamesource, ixnametarget],
            drop=True
        )

    return dfnew


def _return_cartesian_data(X, X_score, showcols, showscores, source_suffix, target_suffix, ixnamepairs):
    if showcols is None:
        showcols = X[0].columns.intersection(X[1].columns)
    X_data = cartesian_join(
        source=X[0][showcols],
        target=X[1][showcols],
        source_suffix=source_suffix,
        target_suffix=target_suffix
    ).set_index(ixnamepairs)
    mycols = list()
    for c in showcols:
        mycols.append(c + '_' + source_suffix)
        mycols.append(c + '_' + target_suffix)
    X_data = X_data[mycols]
    if showscores is not None:
        for c in showscores:
            X_data[c] = X_score[:, c]
    return X_data