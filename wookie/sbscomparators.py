import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, partial_token_set_ratio as tokenratio
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.pipeline import make_union

from wookie import connectors

_navalue_score = None


class BaseSbsComparator(TransformerMixin):
    def __init__(self, on_left='left', on_right='right', compfunc=None, *args, **kwargs):
        """
        base class for all transformers
        Args:
            on_left (str):
            on_right (str):
            compfunc (function): ['fuzzy', 'token', 'exact']
        """
        TransformerMixin.__init__(self)
        self.left = on_left
        self.right = on_right
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc

    def transform(self, X):
        """
        Apply the compfunc to the on_left and on_right column
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        y = X.apply(
            lambda r: self.compfunc(
                r.loc[self.left],
                r.loc[self.right]
            ),
            axis=1
        ).values.reshape(-1, 1)
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
        if not self.on_cols is None:
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
            'name': ['exact', 'fuzzy', 'token'],
            'street': ['exact', 'token'],
            'duns': ['exact'],
            'city': ['fuzzy'],
            'postalcode': ['exact'],
            'country_code':['exact']
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
        self.scoreplan = scoreplan
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self._stages = list()
        self._outcols = list()
        for usedfield in self.scoreplan.keys():
            left = '_'.join([usedfield, 'left'])
            right = '_'.join([usedfield, 'right'])
            for usedscore in self.scoreplan[usedfield]:
                self._stages.append(
                    FuzzyWuzzySbsComparator(on_left=left, on_right=right, comparator=usedscore)
                )
                self._outcols.append('_'.join([usedfield, usedscore]))
        if len(self._stages) > 0:
            self._pipe = make_union(n_jobs=n_jobs, *self._stages, *args, **kwargs)
        else:
            self._pipe = TransformerMixin()
        pass

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


def _evalprecisionrecall(y_true, y_pred):
    """

    Args:
        y_pred (pd.DataFrame/pd.Series): everything that is index is counted as true
        y_true (pd.Series):

    Returns:
        float, float: precision and recall
    """
    true_pos = y_true.loc[y_true > 0]
    true_neg = y_true.loc[y_true == 0]
    # EVERYTHING THAT IS CAUGHT BY Y_PRED IS CONSIDERED AS TRUE
    catched_pos = y_pred.loc[true_pos.index.intersection(y_pred.index)]
    catched_neg = y_pred.loc[y_pred.index.difference(catched_pos.index)]
    missed_pos = true_pos.loc[true_pos.index.difference(y_pred.index)]
    assert true_pos.shape[0] + true_neg.shape[0] == y_true.shape[0]
    assert catched_pos.shape[0] + catched_neg.shape[0] == y_pred.shape[0]
    assert catched_pos.shape[0] + missed_pos.shape[0] == true_pos.shape[0]
    recall = catched_pos.shape[0] / true_pos.shape[0]
    precision = catched_pos.shape[0] / y_pred.shape[0]
    return precision, recall


def _metrics(y_true, y_pred):
    y_pred2 = connectors.indexwithytrue(y_true=y_true, y_pred=y_pred)
    scores = dict()
    scores['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred2)
    scores['precision'] = precision_score(y_true=y_true, y_pred=y_pred2)
    scores['recall'] = recall_score(y_true=y_true, y_pred=y_pred2)
    scores['f1'] = f1_score(y_true=y_true, y_pred=y_pred2)
    return scores


def _evalpred(y_true, y_pred, verbose=True, namesplit=None):
    if namesplit is None:
        sset = ''
    else:
        sset = 'for set {}'.format(namesplit)
    precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=y_pred)
    if verbose:
        print(
            '{} | Pruning score: precision: {:.2%}, recall: {:.2%} {}'.format(
                pd.datetime.now(), precision, recall, sset
            )
        )
    scores = _metrics(y_true=y_true, y_pred=y_pred)
    if verbose:
        print(
            '{} | Model score: precision: {:.2%}, recall: {:.2%} {}'.format(
                pd.datetime.now(), scores['precision'], scores['recall'], sset
            )
        )
    return scores

