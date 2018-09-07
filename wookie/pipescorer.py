import pandas as pd
from sklearn.base import TransformerMixin

from wookie.comparators import FuzzyWuzzyComparator


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


class ScorePlan(dict):
    def _unpack(self):
        outputcols = []
        inputcols = []

        for k in _scorename.keys():
            if self.get(k) is not None:
                for c in self[k]:
                    inputcols.append(c)
                    outputcols.append(c + _scorename[k])
        return inputcols, outputcols

    def compared(self):
        compared_cols = (self._unpack()[0])
        return compared_cols

    def scores(self):
        score_cols = list(self._unpack()[1])
        return score_cols

    def to_dict(self):
        m = dict(
            zip(
                self.keys(),
                self.values()
            )
        )
        for k in self.keys():
            m[k] = self[k]
        return m

    @classmethod
    def from_cols(cls, scorecols):
        """
        Args:
            scorecols (set): list of scoring cols
        Returns:

        """
        x_col = set(scorecols)
        m_dic = {}

        def _findscoreinfo(colname):
            if colname.endswith('_target'):
                k = 'attributes'
                u = _rmv_end_str(colname, '_target')
                return k, u
            elif colname.endswith('_source'):
                k = 'attributes'
                u = _rmv_end_str(colname, '_source')
                return k, u
            elif colname.endswith('score'):
                u = _rmv_end_str(colname, 'score')
                for k in ['fuzzy', 'token', 'exact', 'acronym']:
                    if u.endswith('_' + k):
                        u = _rmv_end_str(u, '_' + k)
                        return k, u
            else:
                return None

        for c in x_col:
            result = _findscoreinfo(c)
            if result is not None:
                method, column = result[0], result[1]
                if m_dic.get(method) is None:
                    m_dic[method] = [column]
                else:
                    m_dic[method] = list(set(m_dic[method] + [column]))
        if len(m_dic) > 0:
            return ScorePlan(m_dic)
        else:
            return None
