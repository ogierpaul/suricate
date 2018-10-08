import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, partial_token_set_ratio as tokenratio
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

from wookie.oaacomparators import _tokencompare


class BaseSbsComparator(TransformerMixin):
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


class DataPasser(TransformerMixin):
    """
    This dont do anything, just pass the data on selected columns
    if on_cols is None, pass the whole dataframe
    """

    def __init__(self, on_cols=None):
        TransformerMixin.__init__(self)
        self.on_cols = on_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        This dont do anything, just pass the data as it is
        Args:
            X:

        Returns:

        """
        if not self.on_cols is None:
            assert isinstance(X, pd.DataFrame)
            assert all(map(lambda c: c in X.columns, self.on_cols))
            res = X[self.on_cols]
        else:
            res = X
        return res


class FuzzyWuzzySbsComparator(BaseSbsComparator, TransformerMixin):
    """
    Compare two columns of a dataframe with one another using functions from fuzzywuzzy library
    """

    def __init__(self, comparator=None, left='left', right='right', *args, **kwargs):
        """
        Args:
            comparator (str): name of the comparator function: ['exact', 'fuzzy', 'token']
            left (str): name of left column
            right (str): name of right column
            *args:
            **kwargs:
        """
        if comparator == 'exact':
            compfunc = _exact_score
        elif comparator == 'fuzzy':
            compfunc = _fuzzy_score
        elif comparator == 'token':
            compfunc = _token_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token']")
        BaseSbsComparator.__init__(
            self,
            compfunc=compfunc,
            left=left,
            right=right,
            *args,
            **kwargs
        )
        pass


class PipeSbsComparator(TransformerMixin):
    """
    Align several FuzzyWuzzyComparator
    Provided that the column are named:
    comp1 = PipeComparator(
    scoreplan={
        'name': ['exact', 'fuzzy', 'token'],
        'street': ['exact', 'token'],
        'duns': ['exact'],
        'city': ['fuzzy'],
        'postalcode': ['exact'],
        'country_code':['exact']
    }
)
    """

    def __init__(self, scoreplan):
        """

        Args:
            scoreplan (dict): of type {'col': 'comparator'}
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
                    FuzzyWuzzySbsComparator(left=left, right=right, comparator=v)
                )
        pipe = make_union(n_jobs=n_jobs, *stages, *args, **kwargs)
        res = pipe.fit_transform(X)

        return res


# Deprecated
class _SbsTokenComparator(TransformerMixin):
    def __init__(self, tokenizer, ixnameleft='ix_left', ixnameright='ix_right', new_col='name', train_col='name'):
        """

        Args:
            tokenizer (TfidfVectorizer): Tokenizer
        """
        TransformerMixin.__init__(self)
        self.tokenizer = tokenizer
        self.new_ix = ixnameleft
        self.train_ix = ixnameright
        self.new_col = new_col
        self.train_col = train_col

    def fit(self, X=None, y=None):
        """
        Do Nothing
        Args:
            X: iterable
            y

        Returns:

        """
        return self

    def transform(self, X):
        """

        Args:
            X (pd.DataFrame):

        Returns:
            pd.DataFrame
        """
        ## format
        new_series = _prepare_deduped_series(X, ix=self.new_ix, val=self.new_col)
        train_series = _prepare_deduped_series(X, ix=self.train_ix, val=self.train_col)

        score = _tokencompare(
            right=train_series,
            left=new_series,
            tokenizer=self.tokenizer,
            ix_left=self.new_ix,
            ix_right=self.train_ix
        )
        score.set_index(
            [self.new_ix, self.train_ix],
            drop=True,
            inplace=True
        )
        score = score.loc[
            X.set_index(
                [self.new_ix, self.train_ix]
            ).index
        ]

        return score


def _prepare_deduped_series(X, ix, val):
    """
    deduplicate the records for one column based on one index column
    Args:
        X (pd.DataFrame)
        ix (str): name of index col
        val (str): name of value col
    Returns:
        pd.Series
    """
    y = X[
        [ix, val]
    ].drop_duplicates(
        subset=[ix]
    ).rename(
        columns={val: 'data'}
    ).set_index(
        ix, drop=True
    ).dropna(
        subset=['data']
    )['data']
    return y


_navalue_score = None


def _valid_inputs(left, right):
    """
    takes two inputs and return True if none of them is null, or False otherwise
    Args:
        left: first object (scalar)
        right: second object (scalar)

    Returns:
        bool
    """
    if any(pd.isnull([left, right])):
        return False
    else:
        return True


def _exact_score(left, right):
    """
    Checks if the two values are equali
    Args:
        left (object): object number 1
        right (object): object number 2

    Returns:
        float
    """
    if _valid_inputs(left, right) is False:
        return _navalue_score
    else:
        return float(left == right)


def _fuzzy_score(left, right):
    """
    return ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if _valid_inputs(left, right) is False:
        return _navalue_score
    else:
        s = (simpleratio(left, right) / 100)
    return s


def _token_score(left, right):
    """
    return the token_set_ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if _valid_inputs(left, right) is False:
        return _navalue_score
    else:
        s = tokenratio(left, right) / 100
    return s
