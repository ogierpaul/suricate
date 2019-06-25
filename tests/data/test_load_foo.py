import pytest
import pandas as pd
from suricate.data.foo import getleft, getright, getXlr, getXsbs, getytrue

def test_left():
    print(getleft())
    assert isinstance(getleft(), pd.DataFrame)
    pass

def test_right():
    print(getright())
    assert isinstance(getright(), pd.DataFrame)
    pass

def test_X_lr():
    print(getXlr())
    assert isinstance(getXlr(), list)
    assert len(getXlr()) == 2
    assert isinstance(getXlr()[0], pd.DataFrame)
    assert isinstance(getXlr()[1], pd.DataFrame)
    pass

def test_X_sbs():
    print(getXsbs())
    assert isinstance(getXsbs(), pd.DataFrame)
    assert getXsbs().shape[0] == getXlr()[0].shape[0] * getXlr()[1].shape[0]
    pass