import pandas as pd
import pytest

n_lines = 100


@pytest.fixture
def df_left():
    left = pd.read_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/left.csv', index_col=0,
                       dtype=str).sample(n_lines)
    return left


@pytest.fixture
def df_right():
    right = pd.read_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/right.csv', index_col=0,
                        dtype=str).sample(n_lines)
    return right
