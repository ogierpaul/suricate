import pandas as pd
import pytest
from sklearn.pipeline import make_union

from wookie.pandasconnectors import VectorizerConnector, ExactConnector
from wookie.preutils import concatixnames

n_lines = 1000


@pytest.fixture
def ix_names():
    ixname = 'ix'
    lsuffix = 'left'
    rsuffix = 'right'
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )
    names = dict()
    names['ixname'] = ixname
    names['ixnameleft'] = ixnameleft
    names['ixnameright'] = ixnameright
    names['ixnamepairs'] = ixnamepairs
    names['lsuffix'] = lsuffix
    names['rsuffix'] = rsuffix
    return names


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


def test_loaddata(ix_names, df_left, df_right):
    print(ix_names['ixname'])
    print(df_left.shape[0])
    print(df_right.shape[0])
    assert True


def test_tfidf(df_left, df_right):
    df_X = [df_left, df_right]
    expected_shape = df_left.shape[0] * df_right.shape[0]
    stages = [
        VectorizerConnector(on='name', analyzer='char', pruning=False),
        VectorizerConnector(on='street', analyzer='char', pruning=False),
        ExactConnector(on='duns', pruning=False)

    ]
    scorer = make_union(*stages)
    X_score = scorer.transform(X=df_X)
    assert X_score.shape[0] == expected_shape
    pass
