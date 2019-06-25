import numpy as np
import pandas as pd
from sklearn.pipeline import make_union

from suricate.lrdftransformers import CartesianLr, ExactConnector, \
    VectorizerConnector, FuzzyConnector, LrDfTransformerMixin, cartesian_join, Indexer, CartesianDataPasser
from suricate.data.base import ix_names
from suricate.data.foo import getleft, getright, getXsbs, getXlr, getytrue
left = getleft()
right = getright()
X_lr = getXlr()
X_sbs = getXsbs()
y_true = getytrue()

def test_fixtures_init():
    print('\n', 'starting test_fixtures_init')
    assert isinstance(ix_names, dict)
    assert isinstance(left, pd.DataFrame)
    assert isinstance(X_sbs, pd.DataFrame)
    print('\n test_fixtures_init successful', '\n\n')


def test_dfconnector():
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
        sbs = connector.show_pairs(y=y, X=X_lr)
        assert isinstance(sbs, pd.DataFrame)
    print('\n test_dfconnector successful', '\n\n')


def test_cartesian_join():
    print('\n', 'starting test_cartesian_join')
    ixname = ix_names['ixname']
    ixnamepairs = ix_names['ixnamepairs']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    df = cartesian_join(left=left, right=right, lsuffix=lsuffix, rsuffix=rsuffix)
    # Output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    # output number of rows is the multiplication of both rows
    assert df.shape[0] == left.shape[0] * right.shape[0]
    # output number of columns are left columns + right columns + 2 columns for each indexes
    assert df.shape[1] == 2 + left.shape[1] + right.shape[1]
    # every column of left and right, + the index, is found with a suffix in the output dataframe
    for oldname in left.reset_index(drop=False).columns:
        newname = '_'.join([oldname, lsuffix])
        assert newname in df.columns
    for oldname in right.reset_index(drop=False).columns:
        newname = '_'.join([oldname, rsuffix])
        assert newname in df.columns

    # assert sidebyside == df
    print('\n test_cartesian_join successful', '\n\n')


def test_cartesian():
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
    y = connector.transform(X=X_lr)
    left = X_lr[0]
    right = X_lr[1]
    assert y.shape[0] == left.shape[0] * right.shape[0]
    assert y.sum() == left.shape[0] * right.shape[0]
    sbs = connector.show_pairs(X=X_lr)
    assert sbs.shape[0] == left.shape[0] * right.shape[0]
    print(sbs)
    print('\n test_cartesian successful', '\n\n')


def test_exact():
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
    y = connector.transform(X=X_lr)
    assert np.nansum(y) == 2
    sbs = connector.show_pairs(X=X_lr)

    connector = ExactConnector(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    score = connector.transform(X=X_lr)
    assert score.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    print('\n test_exact successful', '\n\n')


def test_tfidf():
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
    pairs = connector.transform(X=X_lr)
    assert pairs.shape[0] == 9
    sbs = connector.show_pairs(X=X_lr)
    print(sbs)
    connector.pruning_ths = None
    assert connector.transform(X=X_lr).shape[0] == 9
    print('\n test_tfidf successful', '\n\n')
    pass


def test_makeunion():
    print('\n', 'starting test_makeunion')
    stages = [
        VectorizerConnector(on='name', analyzer='char',
                            ixname=ix_names['ixname'], lsuffix=ix_names['lsuffix'], rsuffix=ix_names['rsuffix']),
        ExactConnector(on='name',
                       ixname=ix_names['ixname'], lsuffix=ix_names['lsuffix'], rsuffix=ix_names['rsuffix'])

    ]
    X_score = make_union(*stages).fit_transform(X=X_lr)
    assert X_score.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    print('\n test_makeunion successful', '\n\n')


def test_fuzzy():
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
    pairs = connector.transform(X=X_lr)
    print(pairs)
    print('\n test_fuzzy successful', '\n\n')


def test_fuzzy_2():
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
    y = y_true
    y = y.loc[y == 1]
    pairs = connector.transform(X=X_lr).flatten()
    print(pairs)
    assert pairs[0] == 1
    assert pairs[4] == 1
    assert pairs[5] > 0
    assert pairs[5] > pairs[7]


def test_indexer():
    con = Indexer(
        on=None,
        ixname=ix_names['ixname'],
        lsuffix=ix_names['lsuffix'],
        rsuffix=ix_names['rsuffix']
    )
    y4 = con.fit_transform(X=X_lr)
    assert y4.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
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
    out = pipe.fit_transform(X=X_lr)
    assert out.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    assert isinstance(out[0][0], tuple)
    assert isinstance(out[0][1], float)
    assert out[0][0] == (0, 0)
    assert out[0][1] == 1.0
    assert out[2][0] == (0, 2)
    assert out[2][1] == 0.0
    assert out[4][0] == (1, 1)
    assert out[4][1] == 1.0
    pass


def test_cartdatapasser():
    dp = CartesianDataPasser()
    out = dp.transform(X=X_lr)
    assert out.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    for c in X_lr[0].columns:
        assert c + '_left' in out.columns
    for c in X_lr[1].columns:
        assert c + '_right' in out.columns
    print(out)
