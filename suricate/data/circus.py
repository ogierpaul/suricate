import pandas as pd

from suricate.dftransformers import cartesian_join
from suricate.preutils.indextools import createmultiindex

def getsource():
    """

    Returns:
        pd.DataFrame: shape (6, 1)
    """
    source = pd.DataFrame(
        {
            'name': [
                'hello world',
                'hello big world',
                'holy grail',
                'moly is holy',
                None,
                'HELLO! world'
            ]
        }
    )
    source.index.name = 'ix'
    return source

def gettarget():
    """
    Identical to getsource but index is changed
    Returns:
        pd.DataFrame: shape (6, 1)
    """
    target = getsource().copy()
    target['ix'] = pd.Series(list('abcdef'), index=target.index)
    target.set_index('ix', drop=True, inplace=True)
    return target

def getXst():
    """

    Returns:
        list: length2 with 2 dataframes, source and target
    """
    X_lr = [getsource(), gettarget()]
    return X_lr

def getXsbs():
    """

    Returns:
        pd.DataFrame: pd.DataFrame: cartesian join of source and target dataframes, shape (36, 2)
    """
    X_st = getXst()
    X_sbs = cartesian_join(source=X_st[0], target=X_st[1])
    return X_sbs

def getytrue():
    """

    Returns:
        pd.Series: supervised training data
    """
    y_true = pd.Series(
        index=createmultiindex(X=getXst()),
        data=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        name='y_true'
    )
    return y_true
