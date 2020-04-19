import pytest
import pandas as pd
from suricate.data.circus import getsource, gettarget, getXst, getXsbs, getytrue

def test_source():
    print(getsource())
    assert isinstance(getsource(), pd.DataFrame)
    assert getsource().shape[0] == 6
    assert getsource().shape[1] == 1

    pass

def test_target():
    print(gettarget())
    assert isinstance(gettarget(), pd.DataFrame)
    assert gettarget().shape[0] == 6
    assert gettarget().shape[1] == 1
    pass

def test_X_lr():
    print(getXst())
    X = getXst()
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