import pytest
import pandas as pd

from suricate.data.base import create_path, open_csv
from suricate.data.companies import getleft, getright, gettrainingdata, _folder_companydf, getytrue

def test_create_path():
    filename = 'left.csv'
    foldername = _folder_companydf
    filepath = create_path(filename=filename, foldername=foldername)
    print(filepath)
    assert isinstance(filepath, str)


def test_load_df():
    filename = 'left.csv'
    foldername = _folder_companydf
    df = open_csv(filename=filename, foldername=foldername)
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)


def test_load_left():
    df = getleft()
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)


def test_load_right():
    df = getright()
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)


def test_load_trainingdata():
    df = gettrainingdata()
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)

def test_load_ytrue():
    y = getytrue()
    print(y.sample(10))
    assert isinstance(y, pd.Series)

