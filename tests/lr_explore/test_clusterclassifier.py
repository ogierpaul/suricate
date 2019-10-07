import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer

from suricate.data.companies import getXlr, getytrue
from suricate.explore import ClusterClassifier
from suricate.explore import SimpleQuestions
from suricate.lrdftransformers import VectorizerConnector, ExactConnector
from suricate.preutils import createmultiindex


@pytest.fixture
def fixture_data():
    # Load the data
    X_lr = getXlr(nrows=300)
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
            ('Scaler', Normalizer())

        ]
    )
    return scores




def test_clusterclassifier(fixture_scores, fixture_data):
    n_clusters = 10
    n_questions = 200
    X_lr = fixture_data
    y_true = getytrue(Xlr=X_lr)
    X_raw = fixture_scores.fit_transform(X=X_lr)
    X_reduced = PCA(n_components=3).fit_transform(X_raw)
    cluster = KMeans(n_clusters=n_clusters)
    y_cluster = pd.Series(
        data=cluster.fit_predict(X=X_reduced),
        index=createmultiindex(X=X_lr)
    )
    questions = SimpleQuestions(n_questions=n_questions)
    ix_questions = questions.fit_transform(X=y_cluster)
    y_true = y_true.loc[y_cluster.index.intersection(y_true.index)]
    print('number of labellized rows found :{}'.format(len(y_true)))
    clf = ClusterClassifier(cluster=cluster)
    clf.fit(X=y_cluster, y=y_true)
    print('all match: {}'.format(clf.allmatch))
    print('no match: {}'.format(clf.nomatch))
    print('mixed match: {}'.format(clf.mixedmatch))
    print('not found: {}'.format(clf.notfound))
    y_pred = clf.predict(X=y_cluster)
    res = pd.Series(y_pred).value_counts()
    print(res)