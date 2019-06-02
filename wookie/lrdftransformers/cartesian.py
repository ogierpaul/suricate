import pandas as pd
from sklearn.base import TransformerMixin

from wookie.lrdftransformers import LrDfTransformerMixin
from wookie.lrdftransformers.base import cartesian_join
from wookie.preutils import concatixnames


class CartesianLr(LrDfTransformerMixin):
    '''
    This transformer returns the cartesian product of left and right indexes
    '''
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', on='all',
                 scoresuffix='cartesianscore', **kwargs):
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix, **kwargs)

    def _transform(self, X, on_ix=None):
        left = X[0]
        right = X[1]
        score = pd.Series(index=self._getindex(X=X, y=on_ix), name=self.outcol).fillna(1)
        return score


class CartesianDataPasser(LrDfTransformerMixin):
    '''
    THIS CLASS IS NOT A DF CONNECTOR BUT A TRANSFORMER MIXIN
    It returns the cartesian join of the two dataframes with all their columns
    '''
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

    def _fit(self, X=None, y=None):
        return self

    def _transform(self, X, y=None):
        return cartesian_join(left=X[0], right=X[1], lsuffix=self.lsuffix, rsuffix=self.rsuffix)
