import numpy as np
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union

from wookie.comparators.sidebyside import FuzzyWuzzySbsComparator
from wookie.preutils import suffixexact, suffixtoken, suffixfuzzy, name_freetext, name_exact, \
    name_usescores


# TODO: Warning: This class might be obsolete / Do not use


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


