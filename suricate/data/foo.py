import pandas as pd
from suricate.data.base import ix_names
from suricate.dftransformers import cartesian_join
from suricate.preutils.indextools import createmultiindex
_samplecol = 'name'


def getsource():
    """

    Returns:
        pd.DataFrame: length (3, 1)
    """
    df = pd.DataFrame(
        {
            ix_names['ixname']: [0, 1, 2],
            _samplecol: ['foo', 'bar', 'ninja']
        }
    ).set_index(
        ix_names['ixname']
    )
    return df

def gettarget():
    """

    Returns:
        pd.DataFrame: length (3, 1)
    """
    df = pd.DataFrame(
        {
            ix_names['ixname']: [0, 1, 2],
            _samplecol: ['foo', 'bar', 'baz']
        }
    ).set_index(
        ix_names['ixname']
    )
    return df

def getXst():
    """

    Returns:
        list: length2 with 2 dataframes, source and target

    """
    X = [getsource(), gettarget()]
    return X

def getXsbs():
    """

    Returns:
        pd.DataFrame: cartesian join of source and target dataframes, shape (9, 2)
    """
    X_sbs = cartesian_join(
        source=getXst()[0],
        target=getXst()[1],
        source_suffix=ix_names['source_suffix'],
        target_suffix=ix_names['target_suffix'],
    ).set_index(
        ix_names['ixnamepairs']
    )
    return X_sbs

def getytrue():
    """

    Returns:
        pd.Series: supervised training data
    """
    y_true = pd.Series(
        data=[1, 0, 0, 0, 1, 1, 0, 0, 0],
        index=createmultiindex(
            X=getXst(),
            names=ix_names['ixnamepairs']
        ),
        name='y_true'
    )
    return y_true