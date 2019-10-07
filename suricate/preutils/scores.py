import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, token_sort_ratio as tokenratio
from geopy.distance import vincenty
from suricate.preutils.metrics import navalue_score


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

def contains_score(left, right):
    """
    check if one string is a substring of another
    Args:
        left (str):
        right (str):

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        if isinstance(left, str) and isinstance(right, str):
            if (left in right) or (right in left):
                return 1.0
            else:
                return 0.0
        else:
            return navalue_score

def vincenty_score(left, right):
    """
    Return vincenty distance
    Args:
        left (tuple): lat lng pair
        right (tuple): lat lng pair

    Returns:
        float
    """
    if left is None or right is None:
        return navalue_score
    else:
        if isinstance(left, tuple) and isinstance(right, tuple):
            return vincenty(left, right)
        else:
            return navalue_score


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