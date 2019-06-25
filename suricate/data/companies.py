import pytest
from suricate.data.base import open_csv

nrows = 100
_folder_companydf = 'csv_company'


def getleft():
    left = open_csv(filename='left.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
    return left

def getright():
    right = open_csv(filename='right.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
    return right

def gettrainingdata():
    training_data = open_csv(filename='trainingdata.csv', foldername=_folder_companydf, index_col=[0,1], nrows=nrows)
    return training_data

def getytrue():
    y_true = gettrainingdata()['y_true']
    return y_true

def getXlr():
    X_lr = [getleft(), getright()]
    return X_lr
