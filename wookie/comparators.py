import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, partial_token_set_ratio as tokenratio
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.pipeline import make_union

from wookie import connectors

_navalue_score = None


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
        y_pred (pd.DataFrame/pd.Series):
        y_true (pd.Series):

    Returns:
        float, float: precision and recall
    """
    true_pos = y_true.loc[y_true > 0]
    true_neg = y_true.loc[y_true == 0]
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

