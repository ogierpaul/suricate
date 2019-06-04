import numpy as np
import pandas as pd
from sklearn.pipeline import make_union

from wookie.lrdftransformers import CartesianLr, ExactConnector, \
    VectorizerConnector, FuzzyConnector, LrDfTransformerMixin, cartesian_join, Indexer, \
    CartesianDataPasser


# from ..data.foo import df_left, df_right, df_X, df_sbs, ix_names

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
    connector = LrDfTransformerMixin(
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
        sbs = connector.show_pairs(y=y, X=df_X)
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
    connector = CartesianLr(
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
    sbs = connector.show_pairs(X=df_X)
    assert sbs.shape[0] == left.shape[0] * right.shape[0]
    print(sbs)
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
    sbs = connector.show_pairs(X=df_X)

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
    sbs = connector.show_pairs(X=df_X)
    print(sbs)
    connector.pruning_ths = None
    assert connector.transform(X=df_X).shape[0] == 9
    assert pairs[0][0] >= 1.0  # 'foo' == 'foo'
    assert pairs[1][0] == 0.0  # 'foo' != 'bar'
    assert pairs[4][0] >= 1.0  # 'bar' == 'bar'
    assert pairs[5][0] > 0 and pairs[5] < 1  # 'bar' ~ 'baz'
    assert pairs[7][0] < pairs[5]
    assert pairs[7][0] > 0  # 'baz'~'bar'> 'baz'~'ninja' > 0
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
    for s in stages:
        output = s.fit_transform(X=df_X)
        assert output.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
        assert output.shape[1] == 1
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
    pairs = connector.transform(X=df_X)
    print(pairs)
    assert pairs[0] == 1  # 'foo' == 'foo'
    assert pairs[1] == 0  # 'foo' != 'bar'
    assert pairs[4] == 1  # 'bar' == 'bar'
    assert pairs[5] > 0 and pairs[5] < 1  # 'bar' ~ 'baz'
    assert pairs[7] < pairs[5] and pairs[7] > 0  # 'baz'~'bar'> 'baz'~'ninja' > 0


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
    for s in stages:
        output = s.fit_transform(X=df_X)
        assert output.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
        assert output.shape[1] == 1
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
    dp = CartesianDataPasser()
    out = dp.transform(X=df_X)
    assert out.shape[0] == df_X[0].shape[0] * df_X[1].shape[0]
    for c in df_X[0].columns:
        assert c + '_left' in out.columns
    for c in df_X[1].columns:
        assert c + '_right' in out.columns
    print(out)
