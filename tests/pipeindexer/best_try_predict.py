import pandas as pd


from suricate.preutils import createmultiindex
from suricate.pipeline.pruningpipe import PruningPipe
from suricate.data.companies import getXst, getytrue
from suricate.explore import Explorer, KBinsCluster
from suricate.dftransformers import DfConnector, VectorizerConnector, ExactConnector
from suricate.sbstransformers import SbsApplyComparator

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score,  balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

_lr_score_list = [
    ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
    ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
    ('name_vecchar', VectorizerConnector(on='name', analyzer='char', ngram_range=(1, 3))),
    ('street_vecchar', VectorizerConnector(on='street', analyzer='char', ngram_range=(1, 3))),
    ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
    ('countrycode_exact', ExactConnector(on='countrycode')),
    ('duns_exact', ExactConnector(on='duns')),
    ('postalcode_exact', ExactConnector(on='postalcode'))

]
_lr_score_cols = [c[0] for c in _lr_score_list]
_sbs_score_list = [
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
    ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
    ('name_token', SbsApplyComparator(on='name', comparator='token')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple')),
    ('postalcode_contains', SbsApplyComparator(on='postalcode', comparator='contains')),
]

def best_try():
    print('start', pd.datetime.now())
    n_rows = None
    n_cluster = 25
    n_simplequestions = 100
    n_hardquestions = 100
    Xst = getXst(nrows=n_rows)
    ixc = createmultiindex(X=Xst)
    y_true = getytrue(Xst=Xst)
    print(pd.datetime.now(), 'data loaded')
    pipe = PruningPipe(
        connector=DfConnector(
            scorer=Pipeline(steps=[
                ('scores', FeatureUnion(_lr_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('pca', PCA(n_components=3))
            ]
            )
        ),
        pruningclf=Explorer(clustermixin=KBinsCluster(n_clusters=n_cluster)),
        sbsmodel=Pipeline(steps=[
                ('scores', FeatureUnion(transformer_list=_sbs_score_list)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('pca', PCA(n_components=3))
            ])
        ,
        classifier=GradientBoostingClassifier(n_estimators=500)
    )
    pipe.fit(X=Xst, y=y_true)
    y_pred = pipe.predict(X=Xst)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    print('***\nscores:\n')
    print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
        precision, recall, accuracy))

if __name__ == '__main__':
    best_try()