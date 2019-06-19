import pytest
import pandas as pd
from suricate.data.foo import left, right, X_lr, X_sbs, y_true

def test_left():
    print(left)
    assert isinstance(left, pd.DataFrame)
    pass

def test_right():
    print(right)
    assert isinstance(right, pd.DataFrame)
    pass

def test_X_lr():
    print(X_lr)
    assert isinstance(X_lr, list)
    assert len(X_lr) == 2
    assert isinstance(X_lr[0], pd.DataFrame)
    assert isinstance(X_lr[1], pd.DataFrame)
    pass

def test_X_sbs():
    print(X_sbs)
    assert isinstance(X_sbs, pd.DataFrame)
    assert X_sbs.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    pass