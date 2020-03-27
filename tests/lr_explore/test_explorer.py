from suricate.data.companies import getXlr, getytrue
from suricate.explore import Explorer, KBinsCluster
from suricate.lrdftransformers import LrDfConnector, VectorizerConnector, ExactConnector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score,  balanced_accuracy_score
import pandas as pd
import numpy as np

nrows = None
_score_list = [
    ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
    ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
    ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
    # ('countrycode_exact', ExactConnector(on='countrycode')),
    ('duns_exact', ExactConnector(on='duns')),
    ('postalcode_exact', ExactConnector(on='postalcode'))

]
_score_cols = [c[0] for c in _score_list]


def test_explorer():
    print(pd.datetime.now())
    n_rows = 200
    n_cluster = 25
    n_simplequestions = 20
    n_hardquestions = 40
    Xlr = getXlr(nrows=n_rows)
    y_true = getytrue(Xlr=Xlr)
    print(pd.datetime.now(), 'data loaded')
    connector = LrDfConnector(
        scorer=Pipeline(
            steps=[
                ('scores', FeatureUnion(_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]
        )
    )
    explorer = Explorer(
        cluster=KBinsCluster(n_clusters=n_cluster),
        n_simple=n_simplequestions,
        n_hard=n_hardquestions
    )
    connector.fit(X=Xlr)
    # Xtc is the transformed output from the connector, i.e. the score matrix
    Xtc = connector.transform(X=Xlr)
    print(pd.datetime.now(), 'score ok')
    # ixc is the index corresponding to the score matrix
    ixc = connector.getindex(X=Xlr)
    ix_simple = explorer.ask_simple(X=Xtc, ix=ixc, fit_cluster=True)
    print(pd.datetime.now(), 'length of ix_simple {}'.format(ix_simple.shape[0]))
    sbs_simple = connector.getsbs(X=Xlr, on_ix=ix_simple)
    print('***** SBS SIMPLE ******')
    print(sbs_simple.sample(5))
    print('*****')
    y_simple = y_true.loc[ix_simple]
    ix_hard = explorer.ask_hard(X=Xtc, y=y_simple, ix=ixc)
    print(pd.datetime.now(), 'length of ix_hard {}'.format(ix_hard.shape[0]))
    sbs_hard = connector.getsbs(X=Xlr, on_ix=ix_hard)
    print(sbs_hard.sample(5))
    print('*****')
    y_train = y_true.loc[ix_simple.union(ix_hard)]
    print('length of y_train: {}'.format(y_train.shape[0]))
    explorer.fit(X=pd.DataFrame(data=Xtc, index=ixc), y=y_train)
    print('results of pred:\n', pd.Series(explorer.predict(X=Xtc)).value_counts())
    print('****')


def test_pruning():
    print('start', pd.datetime.now())
    n_rows = 500
    n_cluster = 25
    n_simplequestions = 50
    n_hardquestions = 50
    Xlr = getXlr(nrows=n_rows)
    y_true = getytrue(Xlr=Xlr)
    print(pd.datetime.now(), 'data loaded')
    connector = LrDfConnector(
        scorer=Pipeline(
            steps=[
                ('scores', FeatureUnion(_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]
        )
    )
    explorer = Explorer(
        cluster=KBinsCluster(n_clusters=n_cluster),
        n_simple = n_simplequestions,
        n_hard=n_hardquestions
    )
    connector.fit(X=Xlr)
    # Xtc is the transformed output from the connector, i.e. the score matrix
    Xtc = connector.transform(X=Xlr)
    print(pd.datetime.now(), 'score ok')
    # ixc is the index corresponding to the score matrix
    ixc = connector.getindex(X=Xlr)
    y_true = y_true.loc[ixc]

    ix_simple = explorer.ask_simple(X=Xtc, ix=ixc, fit_cluster=True)
    ix_hard = explorer.ask_hard(X=Xtc, y=y_true.loc[ix_simple], ix=ixc)
    ix_train = ix_simple.union(ix_hard)
    print('number of training samples:{}'.format(ix_train.shape[0]))
    X_train = pd.DataFrame(data=Xtc, index=ixc).loc[ix_train]
    y_train = y_true.loc[ix_train]

    explorer.fit(X=X_train, y=y_train, fit_cluster=True)
    y_pruning = explorer.predict(X=Xtc)
    y_pruning = pd.Series(data=y_pruning, name='y_pruning', index=ixc)
    y_pred = (y_pruning > 0).astype(int)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    print('***\npruning scores:\n')
    print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
        precision, recall, accuracy))

