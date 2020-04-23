import pytest
from suricate.data.companies import getsource, gettarget, getytrue
import pandas as pd
import  numpy as np
from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_validate
from suricate.dbconnectors import EsConnector
from suricate.dftransformers import ExactConnector, VectorizerConnector, DfConnector
import elasticsearch

_sbs_score_list = [
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
    ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
    ('name_token', SbsApplyComparator(on='name', comparator='token')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple')),
    ('postalcode_contains', SbsApplyComparator(on='postalcode', comparator='contains'))
]

scorer_sbs = FeatureUnion(transformer_list=_sbs_score_list)

from suricate.pipeline import PartialClf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score


pipe = Pipeline(steps=[
    ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
    ('Scaler', Normalizer()),
    ('PCA', PCA(n_components=4)),
    ('Predictor', GradientBoostingClassifier(n_estimators=2000))
])
pred = PartialClf(classifier=pipe)

def test_pipe_es():
    df_source = getsource(nrows=100)
    df_target = gettarget(nrows=None)
    assert df_source.columns.equals(df_target.columns)
    print(pd.datetime.now(),' | ', 'number of rows on left:{}'.format(df_source.shape[0]))
    print(pd.datetime.now(),' | ', 'number of rows on right:{}'.format(df_target.shape[0]))
    esclient = elasticsearch.Elasticsearch()
    scoreplan = {
        'name': {
            'type': 'FreeText'
        },
        'street': {
            'type': 'FreeText'
        },
        'city': {
            'type': 'FreeText'
        },
        'duns': {
            'type': 'Exact'
        },
        'postalcode': {
            'type': 'FreeText'
        },
        'countrycode': {
            'type': 'Exact'
        }
    }
    escon = EsConnector(
        client=esclient,
        scoreplan=scoreplan,
        index="right",
        explain=False,
        size=10
    )
    #Xsm is the similarity matrix
    Xsm = escon.fit_transform(X=df_source)
    ix_con = Xsm.index
    y_true = getytrue(Xst=[df_source, df_target]).loc[ix_con]
    Xsbs = escon.getsbs(X=df_source, on_ix=ix_con)
    scores_further = scorer_sbs.fit_transform(X=Xsbs)
    scores_further = pd.DataFrame(data=scores_further, index=ix_con, columns=[c[0] for c in _sbs_score_list])
    scores_further = pd.concat([Xsm[['es_score']], scores_further], axis=1, ignore_index=False)
    X = scores_further
    scoring = ['precision', 'recall', 'accuracy']
    print(pd.datetime.now(), ' | starting score')
    pipe = Pipeline(steps=[
        ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaler', Normalizer()),
        ('PCA', PCA(n_components=4)),
        ('Predictor', GradientBoostingClassifier(n_estimators=1000, max_depth=5))
    ])

    scores = cross_validate(estimator=pipe, X=X, y=y_true, scoring=scoring, cv=5)
    for c in scoring:
        print(pd.datetime.now(), ' | {} score1: {}'.format(c, np.average(scores['test_'+c])))


def test_pipe_df():
    df_source = getsource(nrows=100)
    df_target = gettarget(nrows=100)
    assert df_source.columns.equals(df_target.columns)
    print(pd.datetime.now(),' | ', 'number of rows on left:{}'.format(df_source.shape[0]))
    print(pd.datetime.now(),' | ', 'number of rows on right:{}'.format(df_target.shape[0]))
    scorer = FeatureUnion(transformer_list=[
        ('name_char', VectorizerConnector(on='name', analyzer='char')),
         ('street_char', VectorizerConnector(on='street', analyzer='char')),
          ('countrycode_exact', ExactConnector(on='countrycode')),
           ('postalcode_exact', ExactConnector(on='postalcode'))
    ])
    dfcon = DfConnector(scorer=scorer)
    Xsm = dfcon.fit_transform(X=[df_source, df_target])
    #TODO: Check column names
    # scorecols = scorer.get_feature_names()
    # print(scorecols)

    ix_con = Xsm.index
    y_true = getytrue(Xst=[df_source, df_target]).loc[ix_con]
    Xsbs = dfcon.getsbs(X=[df_source, df_target], on_ix=ix_con)
    scores_further = scorer_sbs.fit_transform(X=Xsbs)
    scores_further = pd.DataFrame(data=scores_further, index=ix_con, columns=[c[0] for c in _sbs_score_list])
    scores_further = pd.concat([Xsm, scores_further], axis=1, ignore_index=False)
    X = scores_further
    scoring = ['precision', 'recall', 'accuracy']
    print(pd.datetime.now(), ' | starting score')
    pipe = Pipeline(steps=[
        ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaler', Normalizer()),
        ('PCA', PCA(n_components=4)),
        ('Predictor', GradientBoostingClassifier(n_estimators=1000, max_depth=5))
    ])
    scores = cross_validate(estimator=pipe, X=X, y=y_true, scoring=scoring, cv=5)
    for c in scoring:
        print(pd.datetime.now(), ' | {} score1: {}'.format(c, np.average(scores['test_'+c])))



