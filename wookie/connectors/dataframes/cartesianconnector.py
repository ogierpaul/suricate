import pandas as pd

from wookie.connectors.dataframes.base import DFConnector


class CartesianConnector(DFConnector):
    def __init__(self, left, right, ixname='ix', lsuffix='left', rsuffix='right', on='all',
                 scoresuffix='cartesianscore'):
        DFConnector.__init__(self, left=left, right=right, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                             scoresuffix=scoresuffix)

    def fit(self):
        return self

    def transform(self):
        newindex = pd.MultiIndex.from_product([self.leftdf.index, self.rightdf.index], names=self.ixnamepairs)
        score = pd.Series(index=newindex, name=self.outcol).fillna(1)
        return score
