import pandas as pd
import pytest

from wookie.comparators import FuzzyWuzzySbsComparator
from wookie.pandasconnectors import CartesianConnector


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
    cc = CartesianConnector(ixname='ix', lsuffix='left', rsuffix='right', on='name')
    X_sbs = cc.showpairs(X=df_circus)
    return X_sbs


def test_cclr(df_circus, circus_sbs):
    expected_shape = df_circus[0].shape[0] * df_circus[1].shape[0]
    assert circus_sbs.shape[0] == expected_shape
    print(circus_sbs)
    comp = FuzzyWuzzySbsComparator(on_left='name_left', on_right='name_right', comparator='fuzzy')
    X_score = comp.transform(circus_sbs)
    assert X_score.shape[0] == expected_shape
    circus_sbs['score'] = X_score
    print(circus_sbs.sort_values(by='score', ascending=False))
