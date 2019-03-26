import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_union

from wookie.pandasconnectors import CartesianConnector, ExactConnector, \
    VectorizerConnector, FuzzyConnector, DFConnector, cartesian_join, Indexer, CartDataPasser
from wookie.preutils import concatixnames


# This is a test file to review the connectors logic

@pytest.fixture
def ix_names():
    # Create the ix_names variation

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
    names['samplecol'] = 'name'
    return names


@pytest.fixture
def df_left(ix_names):
    """
    initiate a dataframe with just one column (samplecol) and three rows (foo, bar, ninja)
    Returns:
        pd.DataFrame
    """
    samplecol = ix_names['samplecol']
    ixname = ix_names['ixname']
    left = pd.DataFrame(
        {
            ixname: [0, 1, 2],
            samplecol: ['foo', 'bar', 'ninja']
        }
    ).set_index(ixname)
    return left


@pytest.fixture
def df_right(ix_names):
    """
    initiate a dataframe with just one column (samplecol) and three rows (foo, bar, baz)
    Args:
        ix_names:

    Returns:

    """
    samplecol = ix_names['samplecol']
    ixname = ix_names['ixname']
    right = pd.DataFrame(
        {
            ixname: [0, 1, 2],
            samplecol: ['foo', 'bar', 'baz']
        }
    ).set_index(ixname)
    return right


@pytest.fixture
def df_X(df_left, df_right):
    """
    initiate an array with two values, two sample dataframes left and right
    [df_left, df_right]
    Args:
        df_left (pd.DataFrame):
        df_right (pd.DataFrame):

    Returns:
        list
    """
    X = [df_left, df_right]
    return X


@pytest.fixture()
def df_sbs(ix_names):
    """
    Initiate the array of df_left and df_right side-by-side
    Args:
        ix_names:

    Returns:
        pd.DataFrame
    """
    # The side by side should be with None
    ixnameleft = ix_names['ixnameleft']
    ixnameright = ix_names['ixnameright']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    samplecol = ix_names['samplecol']
    sidebyside = pd.DataFrame(
        [
            [0, "foo", 0, "foo", 1],  # case equal
            [0, "foo", 1, "bar", 0],  # case False
            [0, "foo", 2, "baz", 0],
            [1, "bar", 0, "foo", 0],
            [1, "bar", 1, "bar", 1],  # case True for two
            [1, "bar", 2, "baz", 1],  # case True for two fuzzy
            [2, "ninja", 0, "foo", 0],
            [2, "ninja", 1, "bar", 0],
            [2, "ninja", 2, "baz", 0]
            #       [2, "ninja", None]  # case None --> To be removed --> No y_true
        ],
        columns=[ixnameleft, '_'.join([samplecol, lsuffix]), ixnameright, '_'.join([samplecol, rsuffix]), 'y_true']
    )
    sidebyside.set_index(ix_names['ixnamepairs'], inplace=True, drop=True)
    return sidebyside


def test_fixtures_init(ix_names, df_left, df_right, df_sbs):
    print('\n', 'starting test_fixtures_init')
    assert isinstance(ix_names, dict)
    assert isinstance(df_left, pd.DataFrame)
    assert isinstance(df_sbs, pd.DataFrame)
    print('\n test_fixtures_init successful', '\n\n')


def test_dfconnector(ix_names, df_X):
    print('\n', 'starting test_dfconnector')
    ixname = ix_names['ixname']
    ixnamepairs = ix_names['ixnamepairs']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    connector = DFConnector(
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        on='name',
        scoresuffix='levenshtein'
    )
    assert connector.outcol == 'name_levenshtein'
    ## Show side by side
    goodmatches = pd.Series(index=pd.MultiIndex.from_arrays([[0, 1, 1], [0, 1, 2]], names=ixnamepairs),
                            name='y_true').fillna(1)
    for y in goodmatches, pd.DataFrame(goodmatches):
        sbs = connector.showpairs(y=y, X=df_X)
        assert isinstance(sbs, pd.DataFrame)
    print('\n test_dfconnector successful', '\n\n')


def test_cartesian_join(ix_names, df_left, df_right):
    print('\n', 'starting test_cartesian_join')
    ixname = ix_names['ixname']
    ixnamepairs = ix_names['ixnamepairs']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    df = cartesian_join(left=df_left, right=df_right, lsuffix=lsuffix, rsuffix=rsuffix)
    # Output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    # output number of rows is the multiplication of both rows
    assert df.shape[0] == df_left.shape[0] * df_right.shape[0]
    # output number of columns are left columns + right columns + 2 columns for each indexes
    assert df.shape[1] == 2 + df_left.shape[1] + df_right.shape[1]
    # every column of left and right, + the index, is found with a suffix in the output dataframe
    for oldname in df_left.reset_index(drop=False).columns:
        newname = '_'.join([oldname, lsuffix])
        assert newname in df.columns
    for oldname in df_right.reset_index(drop=False).columns:
        newname = '_'.join([oldname, rsuffix])
        assert newname in df.columns

    # assert sidebyside == df
    print('\n test_cartesian_join successful', '\n\n')


def test_cartesian(ix_names, df_X, df_sbs):
    print('\n', 'starting test_cartesian')
    ixname = ix_names['ixname']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    connector = CartesianConnector(
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    ## Show side by side
    y = connector.transform(X=df_X)
    left = df_X[0]
    right = df_X[1]
    assert y.shape[0] == left.shape[0] * right.shape[0]
    assert y.sum() == left.shape[0] * right.shape[0]
    sbs = connector.showpairs(X=df_X)
    assert sbs.shape[0] == left.shape[0] * right.shape[0]
    print(sbs)
    score = connector.pruning_score(X=df_X, y_true=df_sbs['y_true'])
    print(score)
    assert isinstance(score, dict)
    print('\n test_cartesian successful', '\n\n')


def test_exact(ix_names, df_X, df_sbs):
    print('\n', 'starting test_exact')
    ixname = ix_names['ixname']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    connector = ExactConnector(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    ## Show side by side
    y = connector.transform(X=df_X)
    assert np.nansum(y) == 2
    sbs = connector.showpairs(X=df_X)

    score = connector.pruning_score(X=df_X, y_true=df_sbs['y_true'])
    print(score)
    assert isinstance(score, dict)
    connector = ExactConnector(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    score = connector.transform(X=df_X)
    assert score.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
    print('\n test_exact successful', '\n\n')


def test_tfidf(ix_names, df_X):
    print('\n', 'starting test_tfidf')
    ixname = ix_names['ixname']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    connector = VectorizerConnector(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        analyzer='char',
        addvocab='add'
    )
    pairs = connector.transform(X=df_X)
    assert pairs.shape[0] == 9
    sbs = connector.showpairs(X=df_X)
    print(sbs)
    connector.pruning_ths = None
    assert connector.transform(X=df_X).shape[0] == 9
    print('\n test_tfidf successful', '\n\n')
    pass


def test_makeunion(ix_names, df_X):
    print('\n', 'starting test_makeunion')
    stages = [
        VectorizerConnector(on='name', analyzer='char',
                            ixname=ix_names['ixname'], lsuffix=ix_names['lsuffix'], rsuffix=ix_names['rsuffix']),
        ExactConnector(on='name',
                       ixname=ix_names['ixname'], lsuffix=ix_names['lsuffix'], rsuffix=ix_names['rsuffix'])

    ]
    X_score = make_union(*stages).fit_transform(X=df_X)
    assert X_score.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
    print('\n test_makeunion successful', '\n\n')


def test_fuzzy(ix_names, df_X):
    print('\n', 'starting test_fuzzy')
    ixname = ix_names['ixname']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    connector = FuzzyConnector(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    pairs = connector.transform(X=df_X)
    print(pairs)
    print('\n test_fuzzy successful', '\n\n')


def test_fuzzy_2(ix_names, df_X, df_sbs):
    print('\n', 'starting test_fuzzy_2')
    ixname = ix_names['ixname']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    connector = FuzzyConnector(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    y = df_sbs['y_true']
    y = y.loc[y == 1]
    pairs = connector.transform(X=df_X, y=y)
    print(pairs)
    assert pairs[0] == 1
    assert pairs[1] == 1
    assert pairs[2] > 0


def test_indexer(ix_names, df_X, df_sbs):
    con = Indexer(
        on=None,
        ixname=ix_names['ixname'],
        lsuffix=ix_names['lsuffix'],
        rsuffix=ix_names['rsuffix']
    )
    y4 = con.fit_transform(X=df_X)
    assert y4.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
    stages = [
        Indexer(
            ixname=ix_names['ixname'],
            lsuffix=ix_names['lsuffix'],
            rsuffix=ix_names['rsuffix']
        ),
        VectorizerConnector(
            on='name',
            ixname=ix_names['ixname'],
            lsuffix=ix_names['lsuffix'],
            rsuffix=ix_names['rsuffix']
        )
    ]
    pipe = make_union(*stages)
    out = pipe.fit_transform(X=df_X)
    assert out.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
    assert isinstance(out[0][0], tuple)
    assert isinstance(out[0][1], float)
    assert out[0][0] == (0, 0)
    assert out[0][1] == 1.0
    assert out[2][0] == (0, 2)
    assert out[2][1] == 0.0
    assert out[4][0] == (1, 1)
    assert out[4][1] == 1.0
    pass


def test_cartdatapasser(df_X):
    dp = CartDataPasser()
    out = dp.transform(X=df_X)
    assert out.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
    for c in df_X[0].columns:
        assert c + '_left' in out.columns
    for c in df_X[1].columns:
        assert c + '_right' in out.columns
    print(out)
