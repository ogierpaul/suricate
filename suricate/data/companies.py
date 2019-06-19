import pytest
from suricate.data.base import open_csv

nrows = 100
_folder_companydf = 'csv_company'


left = open_csv(filename='left.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
right = open_csv(filename='right.csv', foldername=_folder_companydf, index_col=0, nrows=nrows)
training_data = open_csv(filename='trainingdata.csv', foldername=_folder_companydf, index_col=[0,1], nrows=nrows)
y_true = training_data['y_true']
X_lr = [left, right]
