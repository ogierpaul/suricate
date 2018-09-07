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
