from suricate.data.base import open_csv
from suricate.preutils import createmultiindex
import pandas as pd
from numpy import zeros
nrows = None
_folder_companydf = 'companydata'


def getleft(nrows=nrows):
    left = open_csv(filename='left.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
    return left

def getright(nrows=nrows):
    right = open_csv(filename='right.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
    return right

def gettrainingdata(nrows=nrows):
    training_data = open_csv(filename='trainingdata.csv', foldername=_folder_companydf, index_col=[0,1], nrows=nrows)
    return training_data

def getytrue(Xlr=None):
    """

    Args:
        Xlr: left and right dataframe for which to get the labelling

    Returns:

    """
    if Xlr is None:
        Xlr = getXlr()
    ix_all = createmultiindex(X=Xlr)
    y_true = pd.Series(data=zeros(shape=(ix_all.shape[0],)), index=ix_all, name='y_true').fillna(0)
    y_saved = open_csv(filename='ytrue.csv', foldername=_folder_companydf, index_col=[0, 1])['y_true']
    y_true.loc[y_saved.index.intersection(ix_all)] = y_saved
    return y_true

def getXlr(nrows=nrows):
    X_lr = [getleft(nrows=nrows), getright(nrows=nrows)]
    return X_lr
