import numpy as np
import pandas as pd
from sklearn.pipeline import make_union

from suricate.dftransformers import ExactConnector, \
    VectorizerConnector, DfApplyComparator, DfTransformerMixin, cartesian_join, Indexer, DfVisualSbs
from suricate.data.base import ix_names
from suricate.data.foo import getsource, gettarget, getXsbs, getXst, getytrue
source = getsource()
target = gettarget()
X_lr = getXst()
X_sbs = getXsbs()
y_true = getytrue()

def test_fixtures_init():
    print('\n', 'starting test_fixtures_init')
    assert isinstance(ix_names, dict)
    assert isinstance(source, pd.DataFrame)
    assert isinstance(X_sbs, pd.DataFrame)
    print('\n test_fixtures_init successful', '\n\n')


def test_dfconnector():
    print('\n', 'starting test_dfconnector')
    ixname = ix_names['ixname']
    ixnamepairs = ix_names['ixnamepairs']
    source_suffix = ix_names['source_suffix']
    target_suffix = ix_names['target_suffix']
    connector = DfTransformerMixin(
        ixname=ixname,
        source_suffix=source_suffix,
        target_suffix=target_suffix,
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
    source_suffix = ix_names['source_suffix']
    target_suffix = ix_names['target_suffix']
    usecols = source.columns.intersection(target.columns)
    df = cartesian_join(source=source, target=target, source_suffix=source_suffix, target_suffix=target_suffix)
    # Output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    # output number of rows is the multiplication of both rows
    assert df.shape[0] == source.shape[0] * target.shape[0]
    # output number of columns are left columns + right columns + 2 columns for each indexes
    assert df.shape[1] == source.shape[1] + target.shape[1]
    # every column of source and target,  found with a suffix in the output dataframe
    for oldname in usecols:
        sname = '_'.join([oldname, source_suffix])
        assert sname in df.columns
        tname = '_'.join([oldname, target_suffix])
        assert tname in df.columns

    # assert sidebyside == df
    print('\n test_cartesian_join successful', '\n\n')


def test_exact():
    print('\n', 'starting test_exact')
    ixname = ix_names['ixname']
    source_suffix = ix_names['source_suffix']
    target_suffix = ix_names['target_suffix']
    connector = ExactConnector(
        on='name',
        ixname=ixname,
        source_suffix=source_suffix,
        target_suffix=target_suffix
    )
    ## Show side by side
    y = connector.transform(X=X_lr)
    assert np.nansum(y) == 2
    sbs = connector.show_pairs(X=X_lr)

    connector = ExactConnector(
        on='name',
        ixname=ixname,
        source_suffix=source_suffix,
        target_suffix=target_suffix
    )
    score = connector.transform(X=X_lr)
    assert score.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    print('\n test_exact successful', '\n\n')


def test_tfidf():
    print('\n', 'starting test_tfidf')
    ixname = ix_names['ixname']
    source_suffix = ix_names['source_suffix']
    target_suffix = ix_names['target_suffix']
    connector = VectorizerConnector(
        on='name',
        ixname=ixname,
        source_suffix=source_suffix,
        target_suffix=target_suffix,
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
                            ixname=ix_names['ixname'], source_suffix=ix_names['source_suffix'], target_suffix=ix_names['target_suffix']),
        ExactConnector(on='name',
                       ixname=ix_names['ixname'], source_suffix=ix_names['source_suffix'], target_suffix=ix_names['target_suffix'])

    ]
    X_score = make_union(*stages).fit_transform(X=X_lr)
    assert X_score.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    print('\n test_makeunion successful', '\n\n')


def test_fuzzy():
    print('\n', 'starting test_fuzzy')
    ixname = ix_names['ixname']
    source_suffix = ix_names['source_suffix']
    target_suffix = ix_names['target_suffix']
    connector = DfApplyComparator(
        on='name',
        ixname=ixname,
        source_suffix=source_suffix,
        target_suffix=target_suffix
    )
    pairs = connector.transform(X=X_lr)
    print(pairs)
    print('\n test_fuzzy successful', '\n\n')


def test_fuzzy_2():
    print('\n', 'starting test_fuzzy_2')
    ixname = ix_names['ixname']
    source_suffix = ix_names['source_suffix']
    target_suffix = ix_names['target_suffix']
    connector = DfApplyComparator(
        on='name',
        ixname=ixname,
        source_suffix=source_suffix,
        target_suffix=target_suffix
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
        source_suffix=ix_names['source_suffix'],
        target_suffix=ix_names['target_suffix']
    )
    y4 = con.fit_transform(X=X_lr)
    assert y4.shape[0] == X_lr[0].shape[0] * X_lr[1].shape[0]
    stages = [
        Indexer(
            ixname=ix_names['ixname'],
            source_suffix=ix_names['source_suffix'],
            target_suffix=ix_names['target_suffix']
        ),
        VectorizerConnector(
            on='name',
            ixname=ix_names['ixname'],
            source_suffix=ix_names['source_suffix'],
            target_suffix=ix_names['target_suffix']
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

