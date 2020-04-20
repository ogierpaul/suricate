from suricate.data.companies import getXst, getytrue
from suricate.explore import Explorer, KBinsCluster, cluster_matches, cluster_stats
from suricate.dftransformers import DfConnector, VectorizerConnector, ExactConnector
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
    n_cluster = 10
    n_simplequestions = 200
    n_hardquestions = 200
    Xst = getXst(nrows=n_rows)
    y_true = getytrue(Xst=Xst)
    print(pd.datetime.now(), 'data loaded')
    connector = DfConnector(
        scorer=Pipeline(
            steps=[
                ('scores', FeatureUnion(_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]
        )
    )
    explorer = Explorer(
        clustermixin=KBinsCluster(n_clusters=n_cluster),
        n_simple=n_simplequestions,
        n_hard=n_hardquestions
    )
    connector.fit(X=Xst)
    # Xsm is the transformed output from the connector, i.e. the score matrix
    Xsm = connector.transform(X=Xst)
    print(pd.datetime.now(), 'score ok')
    # ixc is the index corresponding to the score matrix
    ixc = Xsm.index
    ix_simple = explorer.ask_simple(X=pd.DataFrame(data=Xsm, index=ixc), fit_cluster=True)
    print(pd.datetime.now(), 'length of ix_simple {}'.format(ix_simple.shape[0]))
    sbs_simple = connector.getsbs(X=Xst, on_ix=ix_simple)
    print('***** SBS SIMPLE ******')
    print(sbs_simple.sample(5))
    print('*****')
    y_simple = y_true.loc[ix_simple]
    ix_hard = explorer.ask_hard(X=pd.DataFrame(data=Xsm,index=ixc), y=y_simple)
    print(pd.datetime.now(), 'length of ix_hard {}'.format(ix_hard.shape[0]))
    sbs_hard = connector.getsbs(X=Xst, on_ix=ix_hard)
    print(sbs_hard.sample(5))
    print('*****')
    y_train = y_true.loc[ix_simple.union(ix_hard)]
    print('length of y_train: {}'.format(y_train.shape[0]))
    explorer.fit(X=pd.DataFrame(data=Xsm, index=ixc), y=y_train, fit_cluster=True)
    print('results of pred:\n', pd.Series(explorer.predict(X=Xsm)).value_counts())
    print('****')


def test_pruning():
    print('start', pd.datetime.now())
    n_rows = 200
    n_cluster = 10
    n_simplequestions = 200
    n_hardquestions = 200
    Xst = getXst(nrows=n_rows)
    y_true = getytrue(Xst=Xst)
    print(pd.datetime.now(), 'data loaded')
    connector = DfConnector(
        scorer=Pipeline(
            steps=[
                ('scores', FeatureUnion(_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]
        )
    )
    explorer = Explorer(
        clustermixin=KBinsCluster(n_clusters=n_cluster),
        n_simple = n_simplequestions,
        n_hard=n_hardquestions
    )
    connector.fit(X=Xst)
    # Xst is the transformed output from the connector, i.e. the score matrix
    Xsm = connector.transform(X=Xst)
    print(pd.datetime.now(), 'score ok')
    # ixc is the index corresponding to the score matrix
    ixc = Xsm.index
    y_true = y_true.loc[ixc]

    ix_simple = explorer.ask_simple(X=pd.DataFrame(data=Xsm, index=ixc), fit_cluster=True)
    ix_hard = explorer.ask_hard(X=pd.DataFrame(data=Xsm, index=ixc), y=y_true.loc[ix_simple])
    ix_train = ix_simple.union(ix_hard)
    print('number of training samples:{}'.format(ix_train.shape[0]))
    X_train = pd.DataFrame(data=Xsm, index=ixc).loc[ix_train]
    y_train = y_true.loc[ix_train]

    explorer.fit(X=X_train, y=y_train, fit_cluster=True)
    y_pruning = explorer.predict(X=Xsm)
    y_pruning = pd.Series(data=y_pruning, name='y_pruning', index=ixc)
    y_pred = (y_pruning > 0).astype(int)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    print('***\npruning scores:\n')
    print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
        precision, recall, accuracy))

def test_cluster_composition():
    X = pd.DataFrame(
        data={
            'name_score': [0, 0.2, 0.3, 0.5, 0.9, 1.0],
            'street_score': [0.1, 0.25, 0.5, 0.5, 0.8, 0.95],
              },
        index=list('abcdef')
    )
    y_cluster = pd.Series(data=[0, 0, 0, 1, 1, 1], index=list('abcdef'), name='y_cluster')
    y_true = pd.Series(data=[0, 0, 1, 1, 1], index=['a', 'b', 'c', 'e', 'f'], name='y_true')
    print(cluster_matches(y_cluster=y_cluster, y_true=y_true))
    print(cluster_stats(X=X, y_cluster=y_cluster, y_true=y_true))


