import pandas as pd
import pytest
from sklearn.pipeline import make_union, make_pipeline, Pipeline

from wookie.comparators import FuzzyWuzzySbsComparator
from wookie.pandasconnectors import VectorizerConnector, ExactConnector, CartDataPasser
from wookie.preutils import concatixnames

# TODO: this test is to test the possibility of having Left -Right comparators followed by Left-Right Fuzzy comparator

n_lines = 10


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


def test_lrsbs(df_left, df_right):
    df_X = [df_left, df_right]
    expected_shape = df_left.shape[0] * df_right.shape[0]
    lr_stages = [
        VectorizerConnector(on='name', analyzer='char', pruning=False),
        VectorizerConnector(on='street', analyzer='char', pruning=False),
        ExactConnector(on='duns', pruning=False)

    ]
    lr_score = make_union(*lr_stages)

    cc = CartDataPasser()
    assert isinstance(cc.transform(X=df_X), pd.DataFrame)

    sbs_stages = [
        FuzzyWuzzySbsComparator(on_left='name_left', on_right='name_right', comparator='fuzzy'),
        FuzzyWuzzySbsComparator(on_left='street_left', on_right='street_right', comparator='fuzzy'),
        FuzzyWuzzySbsComparator(on_left='duns_left', on_right='duns_right', comparator='exact')
    ]
    sbs_score = make_union(*sbs_stages)

    # BELOW A LOT OF DIFFERENT TEST CASES FOR THE SAME TRANSFORM
    X_score = sbs_score.fit_transform(X=cc.transform(X=df_X))
    assert pd.DataFrame(X_score).shape[1] == len(sbs_stages)

    sbs_pipe = Pipeline([
        ('cartestian', cc),
        ('fuzzyscores', sbs_score)
    ])
    sbs_pipe.fit(X=df_X)
    X_score = sbs_pipe.transform(X=df_X)
    assert pd.DataFrame(X_score).shape[1] == len(sbs_stages)

    X_score = sbs_pipe.fit_transform(X=df_X)
    assert pd.DataFrame(X_score).shape[1] == len(sbs_stages)

    sbs_pipe = make_pipeline(*[cc, sbs_score])
    X_score = sbs_pipe.fit_transform(X=df_X)
    assert pd.DataFrame(X_score).shape[1] == len(sbs_stages)

    allscores = make_union(*[lr_score, sbs_pipe])
    X_score = allscores.fit_transform(X=df_X)
    assert pd.DataFrame(X_score).shape[1] == len(sbs_stages) + len(lr_stages)
    pass
