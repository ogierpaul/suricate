import pandas as pd

from suricate.lrdftransformers import CartesianDataPasser
from suricate.preutils.indextools import createmultiindex

def getleft():
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
    right = getleft().copy()
    right['ix'] = pd.Series(list('abcdef'), index=right.index)
    right.set_index('ix', drop=True, inplace=True)
    return right

def getXlr():
    X_lr = [getleft(), getright()]
    return X_lr

def getXsbs():
    X_sbs = CartesianDataPasser(
        ixname='ix', lsuffix='left', rsuffix='right', on='name'
    ).transform(
        X=getXlr()
    ).set_index(
        ['ix_left', 'ix_right']
    )
    return X_sbs

def getytrue():
    y_true = pd.Series(
        index=createmultiindex(X=getXlr()),
        data=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        name='y_true'
    )
    return y_true
