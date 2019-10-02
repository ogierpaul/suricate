import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion

from localpgsqlconnector import connecttopgsql, pgsqlengine, work_on_ix, select_from_ix
from suricate.data.companies import getXlr
from suricate.preutils import createmultiindex
from tests.lr_pgsqlconnector.notest_setupbase import multiindex21column, _score_list, _score_cols
from suricate.explore.clusterscore import ScoreCluster
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
    n_clusters = 20
    engine = pgsqlengine()
    X_score = pd.read_sql('SELECT * FROM xscore', con=engine).set_index('ix')[_score_cols]
    cluster = ScoreCluster(n_clusters=n_clusters)
    cluster.fit(X=X_score)
    y_cluster = cluster.transform(X_score)
    y_cluster= pd.DataFrame(y_cluster, columns=['y_cluster'], index=X_score.index)
    y_cluster.to_sql('xclusterscore', if_exists='replace', con=engine)
    engine.dispose()
    return True

def analyze_clusters():
    engine = pgsqlengine()
    y_avg = pd.read_sql('SELECT ix, avg_score FROM xscore', con=engine).set_index('ix')
    y_cluster = pd.read_sql('SELECT ix, y_cluster FROM xclusterscore', con=engine).set_index('ix')
    y_true = pd.read_sql('SELECT ix, y_true FROM xpred', con=engine).set_index('ix')
    X = y_avg.join(y_cluster).join(y_true)
    pt = X.pivot_table(index='y_cluster', aggfunc={'y_true': np.mean, 'avg_score':np.mean})
    pt.to_sql('vcluster', con=engine, index=True, if_exists='replace')
    engine.dispose()
    return True



if __name__ == '__main__':
    analyze_clusters()
    pass
