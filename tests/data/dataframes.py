import pandas as pd
import pytest

n_lines = 100

filepath_left = '/Users/paulogier/81-GithubPackages/wookie/operations/data/left.csv'
filepath_right = '/Users/paulogier/81-GithubPackages/wookie/operations/data/right.csv'
filepath_training = '/Users/paulogier/81-GithubPackages/wookie/operations/data/trainingdata.csv'


@pytest.fixture
def df_left():
    left = pd.read_csv(filepath_left, index_col=0,
                       dtype=str)
    return left


@pytest.fixture
def df_right():
    right = pd.read_csv(filepath_right, index_col=0,
                        dtype=str)
    return right


@pytest.fixture
def df_X():
    left = pd.read_csv(filepath_left, index_col=0,
                       dtype=str)
    right = pd.read_csv(filepath_right, index_col=0,
                        dtype=str)
    X = [left, right]
    return X


@pytest.fixture
def y_true():
    df = pd.read_csv(filepath_training, usecols=['ix_left', 'ix_right', 'y_true'])
    y = df.set_index(['ix_left', 'ix_right'])['y_true']
    return y
