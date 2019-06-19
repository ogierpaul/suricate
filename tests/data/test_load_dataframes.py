import pytest
import pandas as pd

from suricate.data.base import create_path, open_csv


def test_create_path():
    filename = 'left.csv'
    foldername = 'datacsv'
    filepath = create_path(filename=filename, foldername=foldername)
    print(filepath)
    assert isinstance(filepath, str)


def test_load_df():
    filename = 'left.csv'
    foldername = 'datacsv'
    filepath = create_path(filename=filename, foldername=foldername)
    df = open_csv(filepath=filepath)
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)


def test_load_left():
    df = open_csv(filepath=create_path(filename='left.csv', foldername='datacsv'), index_col=0)
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)


def test_load_right():
    df = open_csv(filepath=create_path(filename='right.csv', foldername='datacsv'), index_col=0)
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)


def test_load_trainingdata():
    df = open_csv(filepath=create_path(filename='trainingdata.csv', foldername='datacsv'), index_col=[0, 1])
    print(df.sample(10))
    assert isinstance(df, pd.DataFrame)
