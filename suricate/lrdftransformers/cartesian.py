import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion

from suricate.lrdftransformers import LrDfTransformerMixin
from suricate.lrdftransformers.base import cartesian_join
from suricate.preutils import concatixnames, createmultiindex
import itertools

class CartesianLr(LrDfTransformerMixin):
    """
    This transformer returns the cartesian product of left and right indexes
    """

    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', on='all',
                 scoresuffix='cartesianscore', **kwargs):
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix, **kwargs)

    def _transform(self, X):
        """

        Args:
            X (list):

        Returns:
            np.ndarray: transformer returns the cartesian product of left and right indexes \
                of shape(n_samples_left * n_samples_right, 1)
        """
        return np.ones(shape=(X[0].shape[0] * X[1].shape[0], 1))


class CartesianDataPasser(TransformerMixin):
    """
    THIS CLASS IS NOT A DF CONNECTOR BUT A TRANSFORMER MIXIN
    It returns the cartesian join of the two dataframes with all their columns
    """

    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', **kwargs):
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )

    def fit(self, X=None):
        return self

    def transform(self, X, y=None):
        return self._transform(X=X, y=None)

    def _fit(self, X=None, y=None):
        return self

    def _transform(self, X, y=None):
        return cartesian_join(left=X[0], right=X[1], lsuffix=self.lsuffix, rsuffix=self.rsuffix)

class LrDfVisualHelper(TransformerMixin):
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', usecols=None, **kwargs):
        TransformerMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        self.usecols = usecols
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

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):
        """

        Args:
            X (list): [df_left, df_right]
            y: dummy

        Returns:
            pd.DataFrame with index [ix_left, ix_right]
        """
        X_sbs = cartesian_join(left=X[0], right=X[1], lsuffix=self.lsuffix, rsuffix=self.rsuffix)
        # Re-arrange the columns to put the same columns side-by-side
        mycols = [self.ixnameleft, self.ixnameright]
        if self.usecols is None:
            usecols = X[0].columns
        else:
            usecols = self.usecols
        for c in usecols:
            mycols.append(c + '_' + self.lsuffix)
            mycols.append(c + '_' + self.rsuffix)
        X_sbs = X_sbs[mycols].set_index(self.ixnamepairs)
        return X_sbs


def create_lrdf_sbs(X, on_ix=None, ixname = 'ix', lsuffix='left', rsuffix='right'):
    """

    Args:
        X (list): [df_left, df_right]
        on_ix (pd.MultiIndex): collection of pairs ('ix_left', 'ix_right')

    Returns:
        pd.DataFrame, of shape (len(on_ix), df.shape[1] * 2 [{('ix_left', 'ix_right'):('name_left', 'name_right', ...}]
    """
    usecols = X[0].columns.intersection(X[1].columns)
    # Check on_ix is contained into the cartesian join of df_left and df_right
    allix = createmultiindex(X=X, names={'{}_{}'.format(ixname, lsuffix), '{}_{}'.format(ixname, rsuffix)})
    if len(on_ix.difference(allix)) > 0:
        raise IndexError(
            'Indexes called {} not found in cartesian product of left and right dataframe'.format(
                on_ix.difference(allix)
            )
        )

    # Rename the left and right dataframe with suffix
    df_left = X[0][usecols]
    df_left.columns = ['{}_{}'.format(c, lsuffix) for c in usecols]
    df_right = X[1][usecols]
    df_right.columns = [c + '_' + rsuffix for c in usecols]
    # Join on the index
    Xsbs = pd.DataFrame(index=on_ix).reset_index(drop=False)
    Xsbs = Xsbs.join(df_left, on='{}_{}'.format(ixname, lsuffix), how='left')
    Xsbs = Xsbs.join(df_right, on='{}_{}'.format(ixname, rsuffix), how='left')
    # Re-order the columns
    order_cols = [('{}_{}'.format(c, lsuffix), '{}_{}'.format(c, rsuffix)) for c in usecols]
    order_cols = list(itertools.chain(*order_cols))
    Xsbs = Xsbs.set_index(
        ['{}_{}'.format(ixname, lsuffix), '{}_{}'.format(ixname, rsuffix)],
        drop=True
    )
    Xsbs = Xsbs.loc[:, order_cols]
    return Xsbs





class ObsoleteVisualHelper(TransformerMixin):
    """
    Help visualize the scores
    Mix a transformer (FeatureUnion) and usecols data
    """

    def __init__(self, transformer=None, ixname='ix', lsuffix='left', rsuffix='right', **kwargs):
        """

        Args:
            transformer (FeatureUnion):
            ixname:
            lsuffix:
            rsuffix:
            **kwargs:
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
        self.transformer = transformer
        if self.transformer is not None:
            try:
                self.scorecols = [c[1].outcol for c in self.transformer.transformer_list]
            except:
                try:
                    self.scorecols = [c[0] for c in self.transformer.transformer_list]
                except:
                    self.scorecols = None
        else:
            self.scorecols = None

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

    def fit(self, X=None, y=None):
        if self.transformer is not None:
            self.transformer.fit(X=X, y=y)
        return self

    def fit_transform(self, X, y=None, usecols=None, on_ix=None, **fit_params):
        self.fit(X=X, y=y)
        X_out = self.transform(X=X, usecols=usecols, on_ix=None)
        return X_out

    def transform(self, X, usecols=None, on_ix=None):
        """

        Args:
            X:
            usecols:

        Returns:

        """
        if self.transformer is not None:
            X_score = pd.DataFrame(
                data=self.transformer.transform(X),
                index=self._getindex(X=X)
            )
            if self.scorecols is not None:
                X_score.columns = self.scorecols
        else:
            X_score = None

        if usecols is not None:
            X_data = cartesian_join(
                left=X[0][usecols],
                right=X[1][usecols],
                lsuffix=self.lsuffix,
                rsuffix=self.rsuffix
            ).set_index(self.ixnamepairs)
            mycols = list()
            for c in usecols:
                mycols.append(c + '_' + self.lsuffix)
                mycols.append(c + '_' + self.rsuffix)
            X_data = X_data[mycols]
        else:
            X_data = None

        if X_score is not None and X_data is not None:
            X_all = pd.concat([X_data, X_score], axis=1, ignore_index=False)
        elif X_score is None:
            X_all = X_data
        else:
            X_all = X_score

        if on_ix is None:
            return X_all
        else:
            return X_all.loc[on_ix]
