import pytest
import pandas as pd
from suricate.data.circus import getleft, getright, getXlr, getXsbs, getytrue

def test_left():
    print(getleft())
    assert isinstance(getleft(), pd.DataFrame)
    assert getleft().shape[0] == 6
    assert getleft().shape[1] == 1

    pass

def test_right():
    print(getright())
    assert isinstance(getright(), pd.DataFrame)
    assert getright().shape[0] == 6
    assert getright().shape[1] == 1
    pass

def test_X_lr():
    print(getXlr())
    X = getXlr()
    assert isinstance(X, list)
    assert len(X) == 2
    assert isinstance(X[0], pd.DataFrame)
    assert isinstance(X[1], pd.DataFrame)
    assert X[0].shape[0] == 6
    assert X[0].shape[1] == 1
    assert X[1].shape[0] == 6
    assert X[1].shape[1] == 1
    pass

def test_X_sbs():
    print(getXsbs())
    X = getXsbs()
    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] == 36
    assert X.shape[1] == 2
    pass

def test_ytrue():
    print(getytrue())
    X = getytrue()
    assert isinstance(X, pd.Series)
    assert X.shape[0] == 36
    pass