from sklearn.base import TransformerMixin

from wookie.comparators.utils import exact_score, fuzzy_score, token_score


class BaseComparator(TransformerMixin):
    def __init__(self, left='left', right='right', outputCol='outputCol', compfunc=None, *args, **kwargs):
        '''

        Args:
            left (str):
            right (str):
            outputCol (str):
            compfunc (function): ['fuzzy', 'token', 'exact']
        '''
        TransformerMixin.__init__(self)
        self.left = left
        self.right = right
        self.outputCol = outputCol
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc

    def transform(self, X):
        compfunc = self.compfunc
        if not compfunc is None:
            outputCol = self.outputCol
            # noinspection PyCallingNonCallable
            X[outputCol] = X.apply(
                lambda r: compfunc(
                    r.loc[self.left],
                    r.loc[self.right]
                ),
                axis=1
            )
        return X

    def fit(self, *_):
        return self

        # def get_params(self):
        #     params = {
        #         "left": self.left,
        #         "right": self.right,
        #         "compfunc": self.compfunc,
        #         "outputCol": self.outputCol
        #     }
        #     return params
        #
        # def set_params(self, **parameters):
        #     for parameter, value in parameters.items():
        #         setattr(self, parameter, value)
        #     return self


class FuzzyWuzzyComparator(BaseComparator, TransformerMixin):
    def __init__(self, comparator=None, *args, **kwargs):
        if comparator == 'exact':
            compfunc = exact_score
        elif comparator == 'fuzzy':
            compfunc = fuzzy_score
        elif comparator == 'token':
            compfunc = token_score
        else:
            raise ValueError('compfunc value not understood')
        BaseComparator.__init__(self, compfunc=compfunc, *args, **kwargs)
        pass
