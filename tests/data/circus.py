import pandas as pd
import pytest

from wookie.pandasconnectors import CartDataPasser


@pytest.fixture
def df_circus():
    left = pd.DataFrame(
        {
            'name': [
                'hello world',
                'hello big world',
                'holy grail',
                'holy moly'
            ]
        }
    )
    left.index.name = 'ix'
    right = left.copy()
    right['ix'] = pd.Series(['a', 'b', 'c', 'd'], index=right.index)
    right.set_index('ix', drop=True, inplace=True)
    X = [left, right]
    return X


@pytest.fixture
def circus_sbs(df_circus):
    cc = CartDataPasser(ixname='ix', lsuffix='left', rsuffix='right', on='name')
    X_sbs = cc.transform(X=df_circus)
    return X_sbs


@pytest.fixture
def mymatches():
    matches = [
        ['hello world', 'hello big world'],
        ['hello world', 'holy grail'],
        ['hello world', None],
        ['hello world', 'HELLO! world']
    ]
    df = pd.DataFrame(matches)
    df.columns = ['left', 'right']
    return df
