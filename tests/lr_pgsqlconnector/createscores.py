import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion

from localpgsqlconnector import pgsqlengine, work_on_ix, select_from_ix
from suricate.data.companies import getXlr
from suricate.preutils import createmultiindex
from tests.lr_pgsqlconnector.notest_setupbase import multiindex21column, _score_list, _score_cols

# CREATE SCORES
def createscores():
    nrows = None
    Xlr = getXlr(nrows=nrows)
    ix = createmultiindex(X=Xlr)
    ixsimple = multiindex21column(df=pd.DataFrame(index=ix)).index
    scorer = Pipeline(
        steps=[
            ('scores', FeatureUnion(_score_list)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ]
    )
    scores = scorer.fit_transform(X=Xlr)
    score_name = Pipeline(
        steps=[
            ('reduction1d', PCA(n_components=1))
        ]
    ).fit_transform(X=scores[:, :3])
    score_pos = Pipeline(
        steps=[
            ('reduction1d', PCA(n_components=1))
        ]
    ).fit_transform(X=scores[:, 3:])
    score_avg = np.mean(scores, axis=1).reshape(-1, 1)
    X_score = np.hstack([scores, score_name, score_pos, score_avg])
    _scorecols_full = _score_cols + ['name_score', 'pos_score', 'avg_score']
    X_score = pd.DataFrame(data=X_score, index=ixsimple, columns=_scorecols_full)
    # TO SQL
    engine = pgsqlengine()
    X_score.to_sql('xscore', engine, if_exists='replace', index=True)
    engine.dispose()
    return True

def create_clusters():
    from suricate.explore import KBinsCluster
    n_clusters = 25
    engine = pgsqlengine()
    print('loading from sql', pd.datetime.now())
    X_score = pd.read_sql('SELECT * FROM xscore', con=engine).set_index('ix')[_score_cols]
    print('calculating score', pd.datetime.now())
    cluster = KBinsCluster(n_clusters=n_clusters)
    cluster.fit(X=X_score)
    y_cluster = cluster.transform(X_score)
    y_cluster= pd.DataFrame(y_cluster, columns=['y_cluster'], index=X_score.index)
    print('writing to sql', pd.datetime.now())
    start = pd.datetime.now()
    y_cluster.to_sql('xcluster', if_exists='replace', con=engine, chunksize=2000, method='multi')
    engine.dispose()
    print('done', pd.datetime.now())
    end = pd.datetime.now()
    print('loading time: {}'.format((end-start).total_seconds()))
    return True

# def analyze_clusters():
#     engine = pgsqlengine()
#     y_avg = pd.read_sql('SELECT ix, avg_score FROM xscore', con=engine).set_index('ix')
#     y_cluster = pd.read_sql('SELECT ix, y_cluster FROM xcluster', con=engine).set_index('ix')
#     y_true = pd.read_sql('SELECT ix, y_true FROM xpred', con=engine).set_index('ix')
#     X = y_avg.join(y_cluster).join(y_true)
#     pt = X.pivot_table(index='y_cluster', aggfunc={'y_true': np.mean, 'avg_score':np.mean})
#     pt.to_sql('vcluster', con=engine, index=True, if_exists='replace')
#     engine.dispose()
#     return True

def clusterclassifier():
    from suricate.explore import ClusterClassifier, SimpleQuestions, cluster_composition
    engine = pgsqlengine()
    # Take a representative sample from each cluster
    print('loading cluster from sql', pd.datetime.now())
    y_cluster = pd.read_sql('SELECT ix, y_cluster FROM xcluster', con=engine).set_index('ix')['y_cluster']
    sp = SimpleQuestions(n_questions=20)
    sp.fit(X=y_cluster)
    ix_sp = sp.transform(X=y_cluster)
    print('length of questions:{}'.format(ix_sp.shape[0]))
    print('wrinting questions', pd.datetime.now())
    work_on_ix(ix=ix_sp)
    print('fetching y_true', pd.datetime.now())
    y_true = select_from_ix('xtrue').set_index('ix')
    y_true = y_true['y_true']
    print('calculating cluster composition', pd.datetime.now())
    print(y_true.index.intersection(y_cluster.index).shape[0])
    crossdep = cluster_composition(y_cluster=y_cluster, y_true=y_true)
    print('\n********\n')
    print(crossdep)
    print('\n********\n')
    clf = ClusterClassifier()
    clf.fit(X=y_cluster, y=y_true)
    y_pred = clf.predict(X=y_cluster)
    y_pred = pd.Series(y_pred, index=y_cluster.index)
    print(y_pred.value_counts())
    print('writing to sql', pd.datetime.now())
    start = pd.datetime.now()
    pd.DataFrame(y_pred).to_sql('xcluster', if_exists='replace', con=engine, chunksize=2000, method='multi')
    engine.dispose()
    print('done', pd.datetime.now())
    end = pd.datetime.now()
    print('loading time: {}'.format((end-start).total_seconds()))
    return True

def create_pipeline():
    from suricate.explore import KBinsCluster, SimpleQuestions, ClusterClassifier
    nrows = 100
    n_clusters = 25
    n_simple_questions = 20
    Xlr = getXlr(nrows=nrows)
    ix = createmultiindex(X=Xlr)
    ix_all = multiindex21column(df=pd.DataFrame(index=ix)).index
    # work_on_ix(ix=ix_all)
    # X_score = select_from_ix('xscore').set_index('ix')[_score_cols]
    # cluster = KBinsCluster(n_clusters=n_clusters)
    # cluster.fit(X=X_score)
    # y_cluster = cluster.predit(X_score)
    # y_cluster= pd.Series(y_cluster, index=X_score.index)
    # sp = SimpleQuestions(n_questions=20)
    # sp.fit(X=y_cluster)
    # ix_sp = sp.transform(X=y_cluster)
    # work_on_ix(ix=ix_sp)
    # y_true = select_from_ix('xtrue').set_index(['ix'])['y_true']
    # clf = ClusterClassifier()
    # clf.fit(X=y_cluster, y=y_true)
    # y_pred = clf.predict(X=X_score)

    pipetocluster = Pipeline(steps=[
        ('scores', FeatureUnion(_score_list)),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('cluster', KBinsCluster(n_clusters=n_clusters))
    ])
    pipetocluster.fit(X=Xlr)
    y_cluster = pipetocluster.predict(X=Xlr)
    y_cluster = pd.Series(y_cluster.flatten(), index=ix_all)
    work_on_ix(ix_all)
    y_true = select_from_ix('xtrue').set_index('ix')['y_true']
    clf = ClusterClassifier()
    clf.fit(X=y_cluster, y=y_true)
    y_pred = pd.Series(clf.predict(X=y_cluster), index=ix_all)
    print(y_pred.value_counts())

if __name__ == '__main__':
    create_pipeline()
    pass
