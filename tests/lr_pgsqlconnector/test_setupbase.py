import pytest
import numpy as np
from localpgsqlconnector import connecttopgsql, pgsqlengine
import psycopg2
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, QuantileTransformer, FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from suricate.preutils import createmultiindex
from suricate.lrdftransformers import LrDfVisualHelper
from suricate.data.companies import getXlr, getytrue
from suricate.lrdftransformers import VectorizerConnector, ExactConnector, ClusterClassifier
from suricate.pipeline.questions import SimpleQuestions, PointedQuestions
from suricate.lrdftransformers.base import LrDfIndexEncoder
from suricate.pipeline.base import PredtoTrans, TranstoPred

nrows = None
_score_list = [
    ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
    ('name_vecchar', VectorizerConnector(on='name', analyzer='char', ngram_range=(1, 3))),
    ('duns_exact', ExactConnector(on='duns')),
    ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
    ('street_vecchar', VectorizerConnector(on='street', analyzer='char', ngram_range=(1, 3))),
    ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
    ('postalcode_exact', ExactConnector(on='postalcode')),
    ('countrycode_exact', ExactConnector(on='countrycode'))

]
_score_cols = [c[0] for c in _score_list]


def multiindex21column(df):
    df.reset_index(inplace=True, drop=False)
    df['ix'] = df['ix_left'] + '-' + df['ix_right']
    df.set_index('ix', inplace=True, drop=True)
    return df


def create_lrsbs():
    Xlr = getXlr(nrows=nrows)
    ix = createmultiindex(X=Xlr)
    ixsimple = multiindex21column(df=pd.DataFrame(index=ix)).index
    Xsbs = LrDfVisualHelper().fit_transform(X=Xlr)
    Xsbs = multiindex21column(Xsbs)
    engine = pgsqlengine()
    Xsbs.to_sql('xsbs', engine, if_exists='replace', index=True)
    engine.dispose()
    return True


def create_lrscores():
    Xlr = getXlr(nrows=nrows)
    ix = createmultiindex(X=Xlr)
    ixsimple = multiindex21column(df=pd.DataFrame(index=ix)).index
    scorer = Pipeline(
        steps=[
            ('scores', FeatureUnion(_score_list)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]
    )
    scores = scorer.fit_transform(X=Xlr)
    score_name = Pipeline(
        steps=[
            ('Scaler', StandardScaler()),
            ('reduction1d', PCA(n_components=1))
        ]
    ).fit_transform(X=scores[:, :3])
    score_pos = Pipeline(
        steps=[
            ('Scaler', StandardScaler()),
            ('reduction1d', PCA(n_components=1))
        ]
    ).fit_transform(X=scores[:, 3:])
    score_avg = np.mean(scores, axis=1).reshape(-1, 1)

    # Create DataFrame

    X_score = np.hstack([scores, score_name, score_pos, score_avg])
    _scorecols_full = _score_cols + ['name_score', 'pos_score', 'avg_score', 'pca1', 'pca2']
    X_score = pd.DataFrame(data=X_score, index=ixsimple, columns=_scorecols_full)
    # TO SQL
    engine = pgsqlengine()
    X_score.to_sql('xscore', engine, if_exists='replace', index=True)
    engine.dispose()
    return True


def reduce_scores():
    # LOAD DATA
    n_components = 3
    engine = pgsqlengine()
    X_score = pd.read_sql('SELECT * FROM xscore', con=engine).set_index('ix')
    ixsimple = X_score.index
    transfo_pca = PCA(n_components=n_components)
    transfo_pca.fit(X=X_score[_score_cols])
    score_pca = transfo_pca.transform(X=X_score[_score_cols])
    pca_components = transfo_pca.components_
    pca_components = pd.DataFrame(data=pca_components, index=['pca{}'.format(i + 1) for i in range(n_components)],
                                  columns=_score_cols)
    pca_components.to_sql('pca_components', engine, if_exists='replace', index=True)
    score_pca = pd.DataFrame(data=score_pca, index=ixsimple,
                             columns=['pca{}'.format(i + 1) for i in range(n_components)])
    score_pca.to_sql(name='xpca', index=True, con=engine)
    engine.dispose()
    return True


def clustering():
    n_clusters = 12
    # LOAD DATA
    engine = pgsqlengine()
    Xpca = pd.read_sql('SELECT * FROM xpca', con=engine).set_index('ix')
    ixsimple = Xpca.index
    Xcluster = pd.DataFrame(index=ixsimple)
    # CLUSTER PRED
    y_cluster_pca = KMeans(n_clusters=n_clusters).fit_predict(X=Xpca)
    Xcluster['y_cluster'] = y_cluster_pca
    # TO SQL
    Xcluster.to_sql(name='xcluster', index=True, con=engine)
    engine.dispose()
    return True


def simple_questions():
    # LOAD THE DATA
    Xlr = getXlr(nrows=nrows)
    le = LrDfIndexEncoder().fit(X=Xlr)
    engine = pgsqlengine()
    y_cluster = pd.read_sql('SELECT * FROM xcluster', con=engine).set_index('ix')['y_cluster']
    # QUESTIONS
    simplequestions = SimpleQuestions(n_questions=8)
    # ix_questions = le.num_to_ix(vals=simplequestions.fit_transform(X=y_cluster))
    ix_questions = simplequestions.fit_transform(X=y_cluster)
    commonindex = pd.Index(ix_questions).intersection(y_cluster.index)
    assert commonindex.shape[0]> 0
    print(commonindex.shape[0])

    Xsimplequestions = pd.DataFrame(index=y_cluster.index)
    Xsimplequestions['simplequestion'] = False
    Xsimplequestions.loc[ix_questions, 'simplequestion'] = True

    # TO SQL
    Xsimplequestions.to_sql(name='xsimplequestions', index=True, con=engine)
    engine.dispose()
    return True



def y_true():
    # LOAD DATA
    y_true = getytrue()
    assert isinstance(y_true, pd.DataFrame)
    y_true = createmultiindex(y_true).drop[['ix_left', 'ix_right']]
    engine = pgsqlengine()
    y_true.to_sql(name='ytrue', index=True, con=engine)
    engine.dispose()
    return True


def predict():
    # # LOAD DATA
    engine = pgsqlengine()
    engine.dispose()
    # X = pd.read_sql('SELECT * FROM xpca', con=engine).set_index('ix')
    # y = pd.read_sql('SELECT * FROM y_true', con=engine).set_index('ix')['y_true']
    # commonindex = X.index.intersection(y.index)
    # X_train = X.loc[commonindex]
    # y = y.loc[commonindex]
    #
    # # PREDICT
    # rf_classifier = RandomForestClassifier()
    # rf_classifier.fit(X=X_train, y=y)
    # X_pred['y_pred_rf'] = rf_classifier.predict(X=scores)
    # lg_classifier = LogisticRegression()
    # lg_classifier.fit(X=X_score[['avg_score']], y=X_pred['y_true'])
    # X_pred['y_pred_lg'] = lg_classifier.predict(X=X_score[['avg_score']])
    # X_pred['tocheck_rf'] = X_pred['y_true'] != X_pred['y_pred_rf']
    # X_pred['tocheck_lg'] = X_pred['y_true'] != X_pred['y_pred_lg']
    #
    # # TO SQL
    # engine = pgsqlengine()
    # X_pred = multiindex21column(X_pred)
    # X_pred.to_sql('xpred', engine, if_exists='replace', index=True)
    return True

def workflow():
    create_lrsbs()
    create_lrscores()
    reduce_scores()
    clustering()
    simple_questions()
    y_true()
    predict()
    return True

def test_sql():
    simple_questions()
    assert True