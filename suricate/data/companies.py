from suricate.data.base import open_csv
from suricate.preutils import createmultiindex
import pandas as pd
from numpy import zeros
nrows = None
_folder_companydf = 'companydata'


def getsource(nrows=nrows):
    """

    Args:
        nrows (int): number of rows to load

    Returns:
        pd.DataFrame
    """
    df = open_csv(filename='source.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
    return df

def gettarget(nrows=nrows):
    """

    Args:
        nrows(int): number of rows to load

    Returns:
        pd.DataFrame
    """
    df = open_csv(filename='target.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
    return df

def getytrue(Xst=None):
    """

    Args:
        Xst: source and target dataframe for which to get the labelling

    Returns:
        pd.Series: supervised training data
    """
    if Xst is None:
        Xst = getXst()
    ix_all = createmultiindex(X=Xst)
    y_true = pd.Series(data=zeros(shape=(ix_all.shape[0],)), index=ix_all, name='y_true').fillna(0)
    y_saved = open_csv(filename='ytrue.csv', foldername=_folder_companydf, index_col=[0, 1])['y_true']
    y_true.loc[y_saved.index.intersection(ix_all)] = y_saved
    return y_true

def getXst(nrows=nrows):
    """

    Args:
        nrows (int): number of rows to load

    Returns:
        list: length2 with 2 dataframes, source and target
    """
    X = [getsource(nrows=nrows), gettarget(nrows=nrows)]
    return X
