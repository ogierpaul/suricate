import pandas as pd

from wookie.pandasconnectors.base import DFConnector


class Indexer(DFConnector):
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', on=None,
                 scoresuffix=None, **kwargs):
        DFConnector.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                             scoresuffix=scoresuffix, **kwargs)

    def _transform(self, X, on_ix=None):
        ixvals = self._getindex(X=X, y=on_ix)
        ixvals = pd.Series(index=ixvals, data=ixvals.get_values(), name=self.outcol)
        return ixvals
