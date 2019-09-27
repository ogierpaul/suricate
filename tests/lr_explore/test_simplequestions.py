import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Normalizer

from suricate.data.companies import getXlr
from suricate.explore.simplequestions import SimpleQuestions
from suricate.lrdftransformers import VectorizerConnector, ExactConnector, LrDfVisualHelper
from suricate.preutils import createmultiindex



@pytest.fixture
def fixture_data():
    # Load the data
    X_lr = getXlr(nrows=200)
    return X_lr


@pytest.fixture
def fixture_scores():
    # Create the scores
    scores = Pipeline(steps=
        [
            ('scores', FeatureUnion(
                [
                    ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
                    ('name_vecchar', VectorizerConnector(on='name', analyzer='char', ngram_range=(1, 3))),
                    ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
                    ('street_vecchar', VectorizerConnector(on='street', analyzer='char', ngram_range=(1, 3))),
                    ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
                    ('postalcode_exact', ExactConnector(on='postalcode')),
                    ('duns_exact', ExactConnector(on='duns')),
                    ('countrycode_exact', ExactConnector(on='countrycode'))
                ]
            )),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('Scaler', Normalizer()),
            ('reduction3d', PCA(n_components=3))

        ]
    )
    return scores


def test_build_scores(fixture_data, fixture_scores):
    X_lr = fixture_data
    scorer = fixture_scores
    X_score = scorer.fit_transform(X=X_lr)
    assert X_lr[0].shape[0] * X_lr[1].shape[0]== X_score.shape[0]
    assert X_score.shape[1] == 3 # PCA HAS been implemented


def test_build_cluster_2_step(fixture_data, fixture_scores):
    n_clusters = 5
    X_lr = fixture_data
    scorer = fixture_scores
    X_score = scorer.fit_transform(X=X_lr)
    cluster = KMeans(n_clusters=n_clusters)
    y_cluster = cluster.fit_predict(X=X_score)
    assert y_cluster.ndim == 1
    assert np.unique(y_cluster).shape[0] == n_clusters


def test_ask_simple_questions_return_array(fixture_data, fixture_scores):
    n_clusters = 5
    n_questions = 6
    X_lr = fixture_data
    scorer = fixture_scores
    X_score = scorer.fit_transform(X=X_lr)
    cluster = KMeans(n_clusters=n_clusters)
    y_cluster = cluster.fit_predict(X=X_score)
    questions = SimpleQuestions(n_questions=n_questions)
    ix_questions = questions.fit_transform(X=y_cluster)
    # HERE simplequestions is fed via an array, so it returns the row number of each line (it has no index)
    assert ix_questions.ndim == 1
    assert ix_questions.shape[0] <= n_questions * n_clusters
    assert ix_questions.shape[0] > 0


def test_ask_simple_questions_return_multiindex(fixture_data, fixture_scores):
    n_clusters = 5
    n_questions = 6
    X_lr = fixture_data
    scorer = fixture_scores
    X_score = scorer.fit_transform(X=X_lr)
    cluster = KMeans(n_clusters=n_clusters)
    y_cluster = pd.Series(
        data=cluster.fit_predict(X=X_score),
        index=createmultiindex(X=X_lr)
    )
    questions = SimpleQuestions(n_questions=n_questions)
    ix_questions = questions.fit_transform(X=y_cluster)
    assert ix_questions.ndim == 1
    assert ix_questions.shape[0] <= n_questions * n_clusters
    assert ix_questions.shape[0] > 0

    X_sbs = LrDfVisualHelper().fit_transform(X=X_lr)
    X_questions = X_sbs.loc[ix_questions]
    assert X_questions.shape[0] == ix_questions.shape[0]


def test_build_questions_cluster_score(fixture_data, fixture_scores):
    X_lr = fixture_data
    scorer = fixture_scores
    cluster = KMeans(n_clusters=10)
    X_sbs = LrDfVisualHelper().fit_transform(X=X_lr)
    X_score = scorer.fit_transform(X=X_lr)
    y_cluster = cluster.fit_predict(X=X_score)
    y_score = np.mean(X_score, axis=1)
    X_info = pd.DataFrame(data=np.column_stack([y_cluster, y_score]),
                          index=createmultiindex(X=X_lr),
                          columns=['y_cluster', 'y_score'])
    X_all = pd.concat([X_info, X_sbs], ignore_index=False, axis=1)
    X_all.sort_values(by=['y_score', 'y_cluster'], inplace=True)

    questions = SimpleQuestions(n_questions=10)
    ix_questions = questions.fit_transform(X=pd.Series(data=y_cluster, index=createmultiindex(X=X_lr)))
    X_questions = X_all.loc[ix_questions]
    print(X_questions.sample(5))
    assert True


