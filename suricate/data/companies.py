import pytest
from suricate.data.base import open_csv

nrows = 100
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

def getytrue(nrows=nrows):
    y_true = gettrainingdata(nrows=nrows)['y_true']
    return y_true

def getXlr(nrows=nrows):
    X_lr = [getleft(nrows=nrows), getright(nrows=nrows)]
    return X_lr
