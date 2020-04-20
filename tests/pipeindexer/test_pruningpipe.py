from suricate.pipeline.pruningpipe import PruningPipe
from suricate.data.companies import getXst, getytrue
from suricate.explore import Explorer, KBinsCluster
from suricate.dftransformers import DfConnector, VectorizerConnector, ExactConnector
from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score,  balanced_accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from suricate.preutils import createmultiindex

# ESCONNECTOR
from suricate.dbconnectors import EsConnector
import elasticsearch
from suricate.metrics.metrics import get_commonscores

_lr_score_list = [
    ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
    ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
    ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
    ('countrycode_exact', ExactConnector(on='countrycode')),
    ('duns_exact', ExactConnector(on='duns')),
    ('postalcode_exact', ExactConnector(on='postalcode'))

]
_lr_score_cols = [c[0] for c in _lr_score_list]
_sbs_score_list = [
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='fuzzy')),
    ('street_fuzzy', SbsApplyComparator(on='street', comparator='fuzzy')),
    ('name_token', SbsApplyComparator(on='name', comparator='token')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='fuzzy')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='fuzzy')),
    ('postalcode_contains', SbsApplyComparator(on='postalcode', comparator='contains')),
]

def test_pruningpipe():
    print('start', pd.datetime.now())
    n_rows = 500
    n_cluster = 25
    n_simplequestions = 50
    n_pointedquestions = 50
    Xst = getXst(nrows=n_rows)
    ixc = createmultiindex(X=Xst)
    y_true = getytrue()
    y_true = y_true.loc[ixc]
    print(pd.datetime.now(), 'data loaded')
    pipe = PruningPipe(
        connector=DfConnector(
            scorer=Pipeline(steps=[
                ('scores', FeatureUnion(_lr_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))]
            )
        ),
        pruningclf=Explorer(clustermixin=KBinsCluster(n_clusters=n_cluster)),
        sbsmodel=FeatureUnion(transformer_list=_sbs_score_list),
        classifier=LogisticRegressionCV()
    )
    pipe.fit(X=Xst, y=y_true)
    y_pred = pipe.predict(X=Xst)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    print('***\nscores:\n')
    print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
        precision, recall, accuracy))


def test_esconnector():
    print('start', pd.datetime.now())
    n_rows = 500
    n_cluster = 25
    Xst = getXst(nrows=n_rows)
    left = Xst[0]
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
        size=20
    )
    ixc = createmultiindex(X=Xst)
    y_true = getytrue()
    y_true = y_true.loc[ixc]
    print(pd.datetime.now(), 'data loaded')
    pipe = PruningPipe(
        connector=escon,
        pruningclf=Explorer(clustermixin=KBinsCluster(n_clusters=n_cluster)),
        sbsmodel=FeatureUnion(transformer_list=_sbs_score_list),
        classifier=LogisticRegressionCV()
    )
    pipe.fit(X=left, y=y_true)
    y_pred = pipe.predict(X=left)
    scores = get_commonscores(y_pred=y_pred, y_true=y_true)
    precision = scores['precision']
    recall = scores['recall']
    accuracy = scores['balanced_accuracy']
    print('***\nscores:\n')
    print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
        precision, recall, accuracy))
