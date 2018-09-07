import pandas as pd
from fuzzywuzzy.fuzz import ratio, token_set_ratio

navalue_score = 0.5


def valid_inputs(left, right):
    '''
    takes two inputs and return True if none of them is null, or False otherwise
    Args:
        left: first object (scalar)
        right: second object (scalar)

    Returns:
        bool
    '''
    if any(pd.isnull([left, right])):
        return False
    else:
        return True


def exact_score(left, right):
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        return float(left == right)


def fuzzy_score(left, right):
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        s = (ratio(left, right) / 100)
    return s


def token_score(left, right):
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        s = token_set_ratio(left, right) / 100
    return s


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
