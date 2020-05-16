import pandas as pd
from suricate.data.companies import getsource, gettarget, getytrue

from suricate.dbconnectors import EsConnector
from suricate.dftransformers import ExactConnector, VectorizerConnector, DfConnector
import elasticsearch

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score

index_name = 'prime'
doc_type = index_name
marker_col = 'origin'
usecols = ['ix', 'name', 'street', 'city', 'postalcode', 'countrycode', marker_col]

df_source = pd.DataFrame()
df_source[marker_col] = 'new'
df_target = pd.DataFrame()
df_target[marker_col] = 'new'
ssource = getsource()
ssource[marker_col] = 'primer'
starget = gettarget()
starget[marker_col] = 'primer'
sytrue = getytrue()
mix_target = pd.concat([df_target[usecols], starget[usecols]], axis=0, ignore_index=True).reset_index()
mix_source = pd.concat([df_source[usecols], ssource[usecols]], axis=0, ignore_index=True).reset_index()

_sbs_score_list = [
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple'))
]

scorer_sbs = FeatureUnion(transformer_list=_sbs_score_list)
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

# Xsm is the similarity matrix
Xsm = escon.fit_transform(X=mix_source)
ix_con = Xsm.index
y_true = getytrue(Xst=[df_source, df_target]).loc[ix_con]

Xsbs = escon.getsbs(X=mix_source, on_ix=ix_con)
scores_further = scorer_sbs.fit_transform(X=Xsbs)
scores_further = pd.DataFrame(data=scores_further, index=ix_con, columns=[c[0] for c in _sbs_score_list])
scores_further = pd.concat([Xsm[['es_score']], scores_further], axis=1, ignore_index=False)
X = scores_further
pipe = Pipeline(steps=[
    ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
    ('Scaler', Normalizer()),
    ('PCA', PCA(n_components=4)),
    ('Predictor', GradientBoostingClassifier(n_estimators=1000, max_depth=5))
])
#TODO: Get ix_labelled
ix_labelled = y_true.index
#TODO: Get ix_new
ix_new = y_true.index
X_prime = X.loc[ix_labelled]
y_prime = y_true.loc[ix_labelled]
pipe.fit(X=X_prime, y=y_prime)
y_proba = pipe.predict_proba(X=X.loc[ix_new])
y_score = X.loc[ix_new].mean(axis=1)
