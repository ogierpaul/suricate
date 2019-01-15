import pandas as pd

from wookie.connectors.dataframes.base import DFConnector


# TODO: Work in progress

class ExactConnector(DFConnector):
    def __init__(self, left, right, on, ixname='ix', lsuffix='left', rsuffix='right', scoresuffix='exact'):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            on (str): name of column on which to do the pivot
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): name of score suffix to be added at the end
        """
        DFConnector.__init__(self, left=left, right=right, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                             scoresuffix=scoresuffix)
        pass

    def fit(self):
        return self

    def transform(self):


class LrExactComparator(DFConnector):
    def __init__(self,
                 on,
                 scoresuffix='exact',
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 store_threshold=1.0,
                 **kwargs):
        """

        Args:
            on (str): column to compare
            scoresuffix (str): name of the suffix added to the column name for the score name
            ixname: 'ix'
            lsuffix (str): 'left'
            rsuffix (str): 'right'
            store_threshold(flat): variable above which the similarity score is stored
            **kwargs:
        """
        BaseLrComparator.__init__(self,
                                  ixname=ixname,
                                  lsuffix=lsuffix,
                                  rsuffix=rsuffix,
                                  on=on,
                                  scoresuffix=scoresuffix,
                                  store_threshold=store_threshold
                                  )

    def fit(self, left=None, right=None, *args, **kwargs):
        """
        # Do nothing
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):
        Returns:
            self
        """
        return self

    def transform(self, left, right, *args, **kwargs):
        """
        Args:
            left (pd.Series/pd.DataFrame): {'ix':['duns', ...]}
            right (pd.Series/pd.DataFrame):
        Returns:
            pd.Series: {['ix_left', 'ix_right']: 'duns_exact'}
        """
        newleft, newright = self._todf(left=left, right=right)
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
