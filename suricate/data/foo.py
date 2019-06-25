import pandas as pd
from suricate.data.base import ix_names
from suricate.lrdftransformers import cartesian_join
from suricate.preutils.indextools import createmultiindex
_samplecol = 'name'


def getleft():
    left = pd.DataFrame(
        {
            ix_names['ixname']: [0, 1, 2],
            _samplecol: ['foo', 'bar', 'ninja']
        }
    ).set_index(
        ix_names['ixname']
    )
    return left

def getright():
    right = pd.DataFrame(
        {
            ix_names['ixname']: [0, 1, 2],
            _samplecol: ['foo', 'bar', 'baz']
        }
    ).set_index(
        ix_names['ixname']
    )
    return right

def getXlr():
    X_lr = [getleft(), getright()]
    return X_lr

def getXsbs():
    X_sbs = cartesian_join(
        left=getXlr()[0],
        right=getXlr()[1],
        lsuffix=ix_names['lsuffix'],
        rsuffix=ix_names['rsuffix'],
    ).set_index(
        ix_names['ixnamepairs']
    )
    return X_sbs

def getytrue():
    y_true = pd.Series(
        data=[1, 0, 0, 0, 1, 1, 0, 0, 0],
        index=createmultiindex(
            X=getXlr(),
            names=ix_names['ixnamepairs']
        ),
        name='y_true'
    )
    return y_true