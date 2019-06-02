import numpy as np
import pandas as pd

from wookie.lrdftransformers.base import LrDfTransformerMixin


class ExactConnector(LrDfTransformerMixin):
    '''
    This class returns the cartesian product
    '''
    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right', scoresuffix='exact', **kwargs):
        """

        Args:
            on (str): name of column on which to do the pivot
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): name of score suffix to be added at the end
        Returns
            pd.Series
        """
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix, **kwargs)
        pass

    def _transform(self, X, on_ix=None):
        ix = self._getindex(X=X, y=on_ix)
        yleft = X[0][self.on].values
        yright = X[1][self.on].values
        Xcomp = np.transpose([np.repeat(yleft, len(yright)), np.tile(yright, len(yleft))])
        ynp = np.equal(Xcomp[:, 0], Xcomp[:, 1])
        y = pd.Series(index=self._getindex(X=X, y=None), data=ynp)
        y = y.loc[ix]
        return y
