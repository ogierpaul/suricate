import pytest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from suricate.lrdftransformers import LrVisualHelper
from suricate.data.companies import getXlr
from suricate.lrdftransformers import VectorizerConnector, ExactConnector

@pytest.fixture
def fixture_data():
    # Load the data
    X_lr = getXlr(nrows=500)
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

@pytest.fixture
def test_featureunion_scores(fixture_data, fixture_scores):
    X_lr = fixture_data
    scores = fixture_scores
    transformer = FeatureUnion(scores)
    X_score = transformer.fit_transform(X_lr)
    print(X_score.shape)
    assert  True
    return transformer

@pytest.fixture
def test_preprocessing_pipeline(fixture_data, test_featureunion_scores):
    # Impute, scale, and reduce the dimensions
    steps = [
        ('scorer', test_featureunion_scores),
        ('imputer', Imputer(strategy='constant', fill_value=0)),
        ('normalizer_scaler', Normalizer()),
        ('reduction3d', PCA(n_components=3))
    ]
    preprocessing_pipeline = Pipeline(steps)
    X_score_ready = preprocessing_pipeline.fit_transform(X=fixture_data)
    print(X_score_ready.shape)
    return preprocessing_pipeline

def test_cluster1(fixture_data, test_preprocessing_pipeline):
    # Train a basic cluster
    # scores = FeatureUnion(transformer_list=[
    #     ('y_cluster', KMeans(n_clusters=10)),
    #     ('reduction1d', Pipeline(steps=[
    #         ('reduction1d', PCA(n_components=1)),
    #         ('quantile_scaler', QuantileTransformer())
    #     ])),
    #     ('sidebyside', LrVisualHelper())
    # ])
    # y_cluster = KMeans(n_clusters=10).fit_predict(X_score_ready)

