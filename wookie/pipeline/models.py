import pandas as pd
from sklearn.base import TransformerMixin

from wookie.comparators.models import FuzzyWuzzyComparator


# TODO: implement the ScoreDict-Scoreplan routine here

class PipeComparator(TransformerMixin):
    def __init__(self, scoreplan):
        '''

        Args:
            scoreplan (dict):
        '''
        TransformerMixin.__init__(self)
        self.scoreplan = scoreplan

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        '''

        Args:
            X (pd.DataFrame):
            *args:
            **kwargs:

        Returns:

        '''
        stages = []
        for k in self.scoreplan.keys():
            left = '_'.join([k, 'left'])
            right = '_'.join([k, 'right'])
            for v in self.scoreplan[k]:
                outputCol = '_'.join([v, k])
                stages.append(
                    FuzzyWuzzyComparator(left=left, right=right, comparator=v, outputCol=outputCol)
                )
        outputCols = [c.outputCol for c in stages]
        for c in stages:
            X = c.transform(X)
        return X
