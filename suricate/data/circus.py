import pandas as pd

from suricate.lrdftransformers import CartesianDataPasser
from suricate.preutils.indextools import createmultiindex

def getleft():
    """

    Returns:
        pd.DataFrame: shape (6, 1)
    """
    left = pd.DataFrame(
        {
            'name': [
                'hello world',
                'hello big world',
                'holy grail',
                'holy moly',
                None,
                'HELLO! world'
            ]
        }
    )
    left.index.name = 'ix'
    return left

def getright():
    """
    Identical to getleft but index is changed
    Returns:
        pd.DataFrame: shape (6, 1)
    """
    right = getleft().copy()
    right['ix'] = pd.Series(list('abcdef'), index=right.index)
    right.set_index('ix', drop=True, inplace=True)
    return right

def getXlr():
    """

    Returns:
        list: length2 with 2 dataframes, left and right
    """
    X_lr = [getleft(), getright()]
    return X_lr

def getXsbs():
    """

    Returns:
        pd.DataFrame: pd.DataFrame: cartesian join of left and right dataframes, shape (36, 2)
    """
    X_sbs = CartesianDataPasser(
        ixname='ix', lsuffix='left', rsuffix='right', on='name'
    ).transform(
        X=getXlr()
    ).set_index(
        ['ix_left', 'ix_right']
    )
    return X_sbs

def getytrue():
    """

    Returns:
        pd.Series: supervised training data
    """
    y_true = pd.Series(
        index=createmultiindex(X=getXlr()),
        data=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        name='y_true'
    )
    return y_true
