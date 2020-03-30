from suricate.pipeline.pruningpipe import PruningPipe
from suricate.data.companies import getXlr, getytrue
from suricate.explore import Explorer, KBinsCluster
from suricate.lrdftransformers import LrDfConnector, VectorizerConnector, ExactConnector
from suricate.sbsdftransformers import FuncSbsComparator
from suricate.preutils import createmultiindex



import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score,  balanced_accuracy_score
from sklearn.linear_model import LogisticRegressionCV

# Here we define the comparison functions used for the first similarity calculation
_lr_score_list = [
    ('name_vecword', VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2))),
    ('street_vecword', VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2))),
    ('city_vecchar', VectorizerConnector(on='city', analyzer='char', ngram_range=(1, 3))),
    ('countrycode_exact', ExactConnector(on='countrycode')),
    ('duns_exact', ExactConnector(on='duns')),
    ('postalcode_exact', ExactConnector(on='postalcode'))

]
_lr_score_cols = [c[0] for c in _lr_score_list]

# Here we define the comparison functions used for the second similarity calculation
_sbs_score_list = [
    ('name_fuzzy', FuncSbsComparator(on='name', comparator='fuzzy')),
    ('street_fuzzy', FuncSbsComparator(on='street', comparator='fuzzy')),
    ('name_token', FuncSbsComparator(on='name', comparator='token')),
    ('street_token', FuncSbsComparator(on='street', comparator='token')),
    ('city_fuzzy', FuncSbsComparator(on='city', comparator='fuzzy')),
    ('postalcode_fuzzy', FuncSbsComparator(on='postalcode', comparator='fuzzy')),
    ('postalcode_contains', FuncSbsComparator(on='postalcode', comparator='contains')),
]

n_rows = 500 # Number of rows to compare in each datasets
n_cluster = 25 # Number of clusters used in the exploratory step
n_simplequestions = 50 # Number of questions per cluster
n_pointedquestions = 50 # Number of additional questions for clusters with mixed matches


##Load the data
print('start', pd.datetime.now())
Xlr = getXlr(nrows=n_rows)
ixc = createmultiindex(X=Xlr)
# Load the vector corresponding to Xlr
y_true = getytrue().loc[ixc]
print(y_true.value_counts())
print(pd.datetime.now(), 'data loaded')

## Explore the data:
connector = LrDfConnector(
        scorer=Pipeline(steps=[
            ('scores', FeatureUnion(_lr_score_list)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )
    )
explorer = Explorer(clustermixin=KBinsCluster(n_clusters=n_cluster), n_simple=n_simplequestions, n_hard=n_pointedquestions)
explorerpipe.fit_first(Xlr)
simple_questions = explorerpipe.simple_questions(X)
explorerpipe.fit_supervized(Xlr, y_true)
y_true = explorerpipe.hard_questions(X)

## Define the pruning pipe
pipe = PruningPipe(
    connector=connector,
    pruningclf=explorer,
    sbsmodel=FeatureUnion(transformer_list=_sbs_score_list),
    classifier=LogisticRegressionCV()
)
pipe.fit(X=Xlr, y=y_true)

## Predict
y_pred = pipe.predict(X=Xlr)
precision = precision_score(y_true=y_true, y_pred=y_pred)
recall = recall_score(y_true=y_true, y_pred=y_pred)
accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
print('***\nscores:\n')
print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
    precision, recall, accuracy))
