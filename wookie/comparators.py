import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union

from wookie.scoreutils import exact_score, fuzzy_score, token_score


class BaseComparator(TransformerMixin):
    def __init__(self, left='left', right='right', compfunc=None, *args, **kwargs):
        """
        base class for all transformers
        Args:
            left (str):
            right (str):
            compfunc (function): ['fuzzy', 'token', 'exact']
        """
        TransformerMixin.__init__(self)
        self.left = left
        self.right = right
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc

    def transform(self, X):
        """
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        compfunc = self.compfunc
        if not compfunc is None:
            y = X.apply(
                lambda r: compfunc(
                    r.loc[self.left],
                    r.loc[self.right]
                ),
                axis=1
            ).values.reshape(-1, 1)
            return y
        else:
            raise ValueError('compfunc is not defined')


    def fit(self, *_):
        return self


class FuzzyWuzzyComparator(BaseComparator, TransformerMixin):
    def __init__(self, comparator=None, *args, **kwargs):
        if comparator == 'exact':
            compfunc = exact_score
        elif comparator == 'fuzzy':
            compfunc = fuzzy_score
        elif comparator == 'token':
            compfunc = token_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token']")
        BaseComparator.__init__(self, compfunc=compfunc, *args, **kwargs)
        pass


class PipeComparator(TransformerMixin):
    def __init__(self, scoreplan):
        """

        Args:
            scoreplan (dict):
        """
        TransformerMixin.__init__(self)
        self.scoreplan = scoreplan

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, n_jobs=1, *args, **kwargs):
        """

        Args:
            X (pd.DataFrame):
            n_jobs (int): number of jobs
            *args:
            **kwargs:

        Returns:
            np.ndarray
        """
        stages = []
        for k in self.scoreplan.keys():
            left = '_'.join([k, 'left'])
            right = '_'.join([k, 'right'])
            for v in self.scoreplan[k]:
                stages.append(
                    FuzzyWuzzyComparator(left=left, right=right, comparator=v)
                )
        pipe = make_union(n_jobs=n_jobs, *stages, *args, **kwargs)
        res = pipe.fit_transform(X)

        return res
