import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, token_sort_ratio as tokenratio
from sklearn.base import TransformerMixin

from suricate.preutils import navalue_score
from suricate.sbsdftransformers.base import BaseSbsComparator


# TODO: Prio medium : add other comparison functions

class FuncSbsComparator(BaseSbsComparator, TransformerMixin):
    """
    Compare two columns of a dataframe with one another using functions from fuzzywuzzy library
    """

    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right', comparator='fuzzy', *args, **kwargs):
        """
        Args:
            comparator (str): name of the comparator function: ['exact', 'fuzzy', 'token']
            ixname (str): name of the index, default 'ix'
            lsuffix (str): suffix to be added to the left dataframe default 'left', gives --> 'ix_left'
            rsuffix (str): suffix to be added to the left dataframe default 'right', gives --> 'ixright'
            on (str): name of the column on which to do the join
            *args:
            **kwargs:
        """
        if comparator == 'exact':
            compfunc = exact_score
        elif comparator == 'fuzzy':
            compfunc = simple_score
        elif comparator == 'token':
            compfunc = token_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token']")
        BaseSbsComparator.__init__(
            self,
            compfunc=compfunc,
            on_left=on + '_' + lsuffix,
            on_right=on + '_' + rsuffix,
            *args,
            **kwargs
        )
        pass


def simple_score(left, right):
    """
    return ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        s = (simpleratio(left, right) / 100)
    return s


def token_score(left, right):
    """
    return the token_set_ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        s = tokenratio(left, right) / 100
    return s


def exact_score(left, right):
    """
    Checks if the two values are equali
    Args:
        left (object): object number 1
        right (object): object number 2

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        return float(left == right)


def valid_inputs(left, right):
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
