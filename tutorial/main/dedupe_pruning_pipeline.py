from suricate.pipeline.pruningpipe import PruningPipe
from suricate.data.companies import getXst, getytrue
from suricate.explore import Explorer, KBinsCluster
from suricate.dftransformers import DfConnector, VectorizerConnector, ExactConnector
from suricate.sbstransformers import SbsApplyComparator
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
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='fuzzy')),
    ('street_fuzzy', SbsApplyComparator(on='street', comparator='fuzzy')),
    ('name_token', SbsApplyComparator(on='name', comparator='token')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='fuzzy')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='fuzzy')),
    ('postalcode_contains', SbsApplyComparator(on='postalcode', comparator='contains')),
]

n_rows = 500 # Number of rows to compare in each datasets
n_cluster = 10 # Number of clusters used in the exploratory step
n_simplequestions = 100 # Number of questions per cluster
n_pointedquestions = 100 # Number of additional questions for clusters with mixed matches


##Load the data
print('start', pd.datetime.now())
Xst = getXst(nrows=n_rows)
ixc = createmultiindex(X=Xst)
# Load the vector corresponding to Xst
y_true = getytrue().loc[ixc]
print(y_true.value_counts())
print(pd.datetime.now(), 'data loaded')

## Explore the data:
connector = DfConnector(
        scorer=Pipeline(steps=[
            ('scores', FeatureUnion(_lr_score_list)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )
    )
### Fit the cluster non-supervizes
explorer = Explorer(clustermixin=KBinsCluster(n_clusters=n_cluster), n_simple=n_simplequestions, n_hard=n_pointedquestions)
Xst = connector.fit_transform(X=Xst)
explorer.fit_cluster(X=Xst)

### Ask simple questions
ix_simple = explorer.ask_simple(X=Xst)
Sbs_simple = connector.getsbs(X=Xst, on_ix=ix_simple)
y_simple = y_true.loc[ix_simple]

### Fit the cluser with supervized data
explorer.fit(X=Xst, y=y_simple, fit_cluster=False)

### Ask hard (pointed) questions
ix_hard = explorer.ask_hard(X=Xst, y=y_simple)
Sbs_hard = connector.getsbs(X=Xst, on_ix=ix_hard)
y_hard = y_true.loc[ix_hard]

### Obtain the results of the labels
y_questions = y_true.loc[ix_hard.union(ix_simple)]


## Define the pruning pipe
pipe = PruningPipe(
    connector=connector,
    pruningclf=explorer,
    sbsmodel=FeatureUnion(transformer_list=_sbs_score_list),
    classifier=LogisticRegressionCV()
)
pipe.fit(X=Xst, y=y_questions)

## Predict
y_pred = pipe.predict(X=Xst)
precision = precision_score(y_true=y_true, y_pred=y_pred)
recall = recall_score(y_true=y_true, y_pred=y_pred)
accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
print('***\nscores:\n')
print('precision score:{}\n recall score:{}\n balanced accuracy score:{}'.format(
    precision, recall, accuracy))
