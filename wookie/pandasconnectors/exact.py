import pandas as pd

from wookie.pandasconnectors.base import DFConnector


class ExactConnector(DFConnector):
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
        DFConnector.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                             scoresuffix=scoresuffix, **kwargs)
        pass

    def _transform(self, X, on_ix=None):
        newleft, newright = self._todf(left=X[0], right=X[1])
        ix = self._getindex(X=X, y=on_ix)

        score = pd.merge(
            left=newleft.reset_index(drop=False),
            right=newright.reset_index(drop=False),
            left_on=self.on,
            right_on=self.on,
            how='inner',
            suffixes=['_' + self.lsuffix, '_' + self.rsuffix]
        )
        score = score[self.ixnamepairs].set_index(self.ixnamepairs)
        score[self.outcol] = 1
        score = score[self.outcol]
        return score
