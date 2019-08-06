import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from suricate.lrdftransformers import LrVisualHelper
from suricate.data.companies import getXlr
from suricate.lrdftransformers import VectorizerConnector, ExactConnector
from suricate.pipeline.questions import SimpleQuestions

@pytest.fixture
def fixture_data():
    # Load the data
    X_lr = getXlr(nrows=100)
    return X_lr


@pytest.fixture
def fixture_scores():
    # Create the scores
    scores = [
        ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
        ('name_vecchar', VectorizerConnector(on='name', analyzer='char', ngram_range=(1, 3))),
        ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
        ('street_vecchar', VectorizerConnector(on='street', analyzer='char', ngram_range=(1, 3))),
        ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
        ('postalcode_exact', ExactConnector(on='postalcode')),
        ('duns_exact', ExactConnector(on='duns')),
        ('countrycode_exact', ExactConnector(on='countrycode'))
    ]
    return scores


def test_featureunion_scores(fixture_data, fixture_scores):
    X_lr = fixture_data
    scores = fixture_scores
    transformer = FeatureUnion(scores)
    X_score_raw = transformer.fit_transform(X_lr)
    print(X_score_raw.shape)

    # Impute, scale, and reduce the dimensions

    reduce1d = Pipeline(
        steps=[
            ('imputer', Imputer()),
            ('scaler_quantile', QuantileTransformer()),
            ('reduction1d', PCA(n_components=1))
        ]
    )
    reduce3d = Pipeline(steps=[
        ('imputer', Imputer(strategy='constant', fill_value=0)),
        ('normalizer_scaler', Normalizer()),
        ('reduction3d', PCA(n_components=3))
    ]
    )
    X_score3d = reduce3d.fit_transform(X=X_score_raw)
    X_score1d = reduce1d.fit_transform(X=X_score_raw)
    y_cluster = KMeans(n_clusters=10).fit_predict(X=X_score3d).reshape(-1, 1)
    X_all = np.hstack([X_score1d, y_cluster])
    X_sbs = LrVisualHelper().fit_transform(X=X_lr)
    questions = SimpleQuestions(n_questions=10)
    questions.fit(X=y_cluster)
    print(y_cluster.ndim)
    y_questions = questions.transform(y_cluster)
    print(y_questions)







