import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union

from wookie.preutils import suffixexact, suffixtoken, suffixfuzzy, name_freetext, name_exact, \
    exact_score, fuzzy_score, token_score, name_usescores


class BaseSbsComparator(TransformerMixin):
    def __init__(self, on_left='left', on_right='right', compfunc=None, n_jobs=1, *args, **kwargs):
        """
        base class for all transformers
        Args:
            on_left (str):
            on_right (str):
            compfunc (callable): ['fuzzy', 'token', 'exact']
            n_jobs (int): number of parallel jobs. If n_jobs>1, will call the dask dataframe API
        """
        TransformerMixin.__init__(self)
        self.left = on_left
        self.right = on_right
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc
        self.n_jobs = n_jobs

    def transform(self, X):
        """
        Apply the compfunc to the on_left and on_right column
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        if self.n_jobs == 1:
            return self._stransform(X)
        else:
            return self._ptransform(X)

    def _stransform(self, X):
        """
        Pandas apply function on rows (axis=1).
        Args:
            X (pd.DataFrame):

        Returns:
            np.array
        """
        y = X.apply(
            lambda r: self.compfunc(
                r.loc[self.left],
                r.loc[self.right]
            ),
            axis=1
        ).values.reshape(-1, 1)
        return y

    def _ptransform(self, X):
        """
        Use of dask dataframe API to parall
        Args:
            X:

        Returns:

        """
        y = dd.from_pandas(
            X.reset_index(drop=True),
            npartitions=self.n_jobs
        ).map_partitions(
            func=self.transform
        )
        return y

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

    def fit(self, X=None, y=None):

        return self

    def transform(self, X):
        """
        This dont do anything, just pass the data as it is
        Args:
            X:

        Returns:

        """
        if self.on_cols is not None:
            assert isinstance(X, pd.DataFrame)
            res = X[self.on_cols]
        else:
            res = X
        return res


class FuzzyWuzzySbsComparator(BaseSbsComparator, TransformerMixin):
    """
    Compare two columns of a dataframe with one another using functions from fuzzywuzzy library
    """

    def __init__(self, on_left, on_right, comparator=None, *args, **kwargs):
        """
        Args:
            comparator (str): name of the comparator function: ['exact', 'fuzzy', 'token']
            on_left (str): name of left column
            on_right (str): name of right column
            *args:
            **kwargs:
        """
        if comparator == 'exact':
            compfunc = exact_score
        elif comparator == 'fuzzy':
            compfunc = fuzzy_score
        elif comparator == 'token':
            compfunc = token_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token']")
        BaseSbsComparator.__init__(
            self,
            compfunc=compfunc,
            on_left=on_left,
            on_right=on_right,
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
            'name': {
                'type': 'FreeText',
                'use_scores': ['fuzzy', 'token']
            }
            'country_code': {
                'type': 'Exact',
                'use_scores': ['exact']
            }
        }
    )
    if no scoreplan is passed, (empty dict), returns an empty array
    """

    def __init__(self, scoreplan, lsuffix='left', rsuffix='right', n_jobs=1, *args, **kwargs):
        """

        Args:
            scoreplan (dict): of type {'col': 'comparator'}
            lsuffix (str): 'left'
            rsuffix (str): 'right'
            n_jobs (int)
        """
        TransformerMixin.__init__(self)
        assert isinstance(scoreplan, dict)
        self.scoreplan = scoreplan.copy()
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self._stages = list()
        self.outcols = list()
        self.usecols = list()
        for inputfield in self.scoreplan.keys():
            use_scores = self._pick_scores(usedfield=inputfield)
            if use_scores is not None:
                left = '_'.join([inputfield, self.lsuffix])
                right = '_'.join([inputfield, self.rsuffix])
                for used_score in use_scores:
                    if used_score in [suffixexact, suffixtoken, suffixfuzzy]:
                        self._stages.append(
                            FuzzyWuzzySbsComparator(on_left=left, on_right=right, comparator=used_score)
                        )
                        self.outcols.append('_'.join([inputfield, used_score]))
                        if not inputfield in self.usecols:
                            self.usecols.append(inputfield)
        if len(self._stages) > 0:
            self._pipe = make_union(n_jobs=n_jobs, *self._stages, *args, **kwargs)
        else:
            self._pipe = TransformerMixin()
        pass

    def _pick_scores(self, usedfield):
        """
        Extract the possible scores from the score plan, giving default if needed
        Args:
            usedfield (str):

        Returns:
            list
        """
        # Create the score plan
        use_scores = None
        if self.scoreplan[usedfield]['type'] == name_exact:
            use_scores = [suffixexact]
        elif self.scoreplan[usedfield]['type'] == name_freetext:
            if self.scoreplan[usedfield].get(name_usescores) is not None:
                use_scores = self.scoreplan[usedfield].get(name_usescores)
                use_scores = list(
                    filter(
                        lambda r: r in [suffixfuzzy, suffixtoken, suffixtoken],
                        use_scores
                    )
                )
            if use_scores is None:
                use_scores = [suffixtoken, suffixfuzzy]
        return use_scores

    def fit(self, *args, **kwargs):
        """
        Do nothing
        Args:
            *args:
            **kwargs:

        Returns:

        """
        return self

    def transform(self, X, *args, **kwargs):
        """
        Transform method
        if no score plan passed return empty array
        Args:
            X (pd.DataFrame):
            *args:
            **kwargs:

        Returns:
            np.ndarray
        """
        if len(self._stages) > 0:
            res = self._pipe.fit_transform(X)
        else:
            # if no score plan passed return empty array
            res = np.zeros(shape=(X.shape[0], 0))
        return res


