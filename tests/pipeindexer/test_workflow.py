import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, QuantileTransformer, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from suricate.preutils import createmultiindex
from suricate.lrdftransformers import LrVisualHelper
from suricate.data.companies import getXlr, getytrue
from suricate.lrdftransformers import VectorizerConnector, ExactConnector, ClusterClassifier
from suricate.pipeline.questions import SimpleQuestions, PointedQuestions
from suricate.lrdftransformers.base import LrDfIndexEncoder
from suricate.pipeline.base import PredtoTrans, TranstoPred

def similarity_score(X):
    return np.mean(X, axis=1)
similarity = Pipeline(
    steps=[
        ('reduction1d', FunctionTransformer(func=similarity_score))
    ]
)

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

def test_build_sbsinfo(fixture_data, fixture_scores):
    X_lr = fixture_data
    scorer = fixture_scores
    cluster = KMeans(n_clusters=10)
    le = LrDfIndexEncoder().fit(X=X_lr)
    X_sbs = LrVisualHelper().fit_transform(X=X_lr)
    X_score = scorer.fit_transform(X=X_lr)
    y_cluster = PredtoTrans(estimator=cluster).fit_transform(X=X_score)
    y_score = similarity.fit_transform(X=X_score)
    X_info = pd.DataFrame(data=np.column_stack([y_cluster, y_score]),
                          index=createmultiindex(X=X_lr),
                          columns=['y_cluster', 'y_score'])
    X_all = pd.concat([X_info, X_sbs], ignore_index=False, axis=1)
    assert X_all.shape[0] == X_score.shape[0]


def test_build_simplequestions(fixture_data, fixture_scores):
    X_lr = fixture_data
    scorer = fixture_scores
    cluster = KMeans(n_clusters=10)
    le = LrDfIndexEncoder().fit(X=X_lr)
    X_sbs = LrVisualHelper().fit_transform(X=X_lr)
    X_score = scorer.fit_transform(X=X_lr)
    y_cluster = PredtoTrans(estimator=cluster).fit_transform(X=X_score)
    y_score = similarity.fit_transform(X=X_score)
    X_info = pd.DataFrame(data=np.column_stack([y_cluster, y_score]),
                          index=createmultiindex(X=X_lr),
                          columns=['y_cluster', 'y_score'])
    X_all = pd.concat([X_info, X_sbs], ignore_index=False, axis=1)


    simplequestions = SimpleQuestions(n_questions=10)
    ix_questions = le.num_to_ix(vals=simplequestions.fit_transform(X=y_cluster))
    X_questions = X_all.loc[ix_questions]
    X_questions.sort_values(by=['y_score', 'y_cluster'], ascending=False, inplace=True)
    print(X_questions.head(5))
    assert True


def test_build_hard_questions(fixture_data, fixture_scores):
    X_lr = fixture_data
    scorer = fixture_scores
    cluster = KMeans(n_clusters=10)
    le = LrDfIndexEncoder().fit(X=X_lr)
    X_sbs = LrVisualHelper().fit_transform(X=X_lr)
    X_score = scorer.fit_transform(X=X_lr)
    y_cluster = PredtoTrans(estimator=cluster).fit_transform(X=X_score)
    y_score = similarity.fit_transform(X=X_score)
    X_info = pd.DataFrame(data=np.column_stack([y_cluster, y_score]),
                          index=createmultiindex(X=X_lr),
                          columns=['y_cluster', 'y_score'])
    X_all = pd.concat([X_info, X_sbs], ignore_index=False, axis=1)

    simplequestions = SimpleQuestions(n_questions=10)
    ix_questions = le.num_to_ix(vals=simplequestions.fit_transform(X=y_cluster))
    y_true = getytrue(nrows=None)


    cluspred = ClusterClassifier()
    y_cluster = pd.Series(data=y_cluster, index=createmultiindex(X=X_lr))
    ix_common = pd.Index(data=ix_questions.values).intersection(y_true.index)
    y_true = y_true.loc[ix_common]
    y_pred_cluster = cluspred.fit_predict(X=y_cluster, y=y_true)
    questions_hard = PointedQuestions(n_questions=10).fit(X=y_cluster, y=y_pred_cluster)
    ix_hard_questions = questions_hard.transform(X=y_cluster)
    X_hard_questions = pd.DataFrame(data=X_all[ix_hard_questions]).sort_values(by=0, ascending=False)
