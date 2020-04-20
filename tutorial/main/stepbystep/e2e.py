# End to End Dedupe
## Pruning pipeline using Elastic search

from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
from suricate.data.companies import getsource, gettarget
import pandas as pd
import  numpy as np

engine = create_engine_ready()

# filefolder = '~/'
# leftpath = 'source.csv'
# rightpath = 'target.csv'
# df_source = pd.read_csv(filefolder + leftpath, index_col=0, sep='|', encoding='utf-8')
# df_target = pd.read_csv(filefolder + rightpath, index_col=0, sep='|', encoding='utf-8')
df_source_raw = getsource(nrows=100)
df_target_raw = gettarget(nrows=100)

def rebuild_ytrue(ix):
    y_true_saved = pd.read_sql(sql="SELECT * FROM y_true WHERE y_true.y_true = 1", con=engine).set_index(
        ['ix_source', 'ix_target'],
        drop=True)['y_true']
    y = pd.Series(index=ix, data = np.zeros(shape=len(ix)), name='y_true')
    ix_common = y_true_saved.index.intersection(ix)
    y.loc[ix_common] = y_true_saved.loc[ix_common]
    return y



def prepare_source(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2


def prepare_target(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2


df_source = prepare_source(df_source_raw)
df_target = prepare_target(df_target_raw)
assert df_source.columns.equals(df_target.columns)
print(pd.datetime.now(),' | ', 'number of rows on left:{}'.format(df_source.shape[0]))
print(pd.datetime.now(),' | ', 'number of rows on right:{}'.format(df_target.shape[0]))


## Push the data to ES

import elasticsearch
import pandas as pd
from suricate.dbconnectors.esconnector import index_with_es
import time


## Put the data to ES, drop the index first and then re create
esclient = elasticsearch.Elasticsearch()
es_indice = 'df_target'

print(pd.datetime.now(),' | ', 'Start pushing to ES')

if True:
    try:
        esclient.indices.delete(index=es_indice)
    except:
        pass
    request_body = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 5
        },

        "mappings": {
            "_doc": {
                "properties": {
                    "ix": {"type": "keyword"},
                    "name": {"type": "text"},
                    "street": {"type": "text"},
                    "city": {"type": "text"},
                    "postalcode": {"type": "text"},
                    "countrycode": {"type": "keyword"}
                }
            }
        }
    }
    esclient.indices.create(index=es_indice, body=request_body)
    index_with_es(client=esclient, df=df_target, index=es_indice, ixname="ix", reset_index=True, doc_type='_doc')
    time.sleep(5)
pass
catcount = esclient.count(index=es_indice)['count']
assert catcount == df_target.shape[0]
print(pd.datetime.now(),' | ', 'pushed to es_sql indice {}'.format(es_indice))
print(pd.datetime.now(),' | ', 'number of docs: {}'.format(catcount))


## Connect the data
from tutorial.main.stepbystep.stepbysteputils.esconnector import getesconnector

print(pd.datetime.now(),' | ', 'Starting connection')
escon = getesconnector()

Xst = escon.fit_transform(X=df_source)
print(pd.datetime.now(),' | ', 'Finished connection')
print(pd.datetime.now(),' | ', 'number of pairs {}'.format(Xst.shape[0]))
print(pd.datetime.now(),' | ', 'Connection scores sample:')
print(Xst.sample(5))

ix_con_multi = Xst.index
print(pd.datetime.now(),' | ', 'Starting side-by-side build')
Xsbs = escon.getsbs(X=df_source, on_ix=ix_con_multi)
print(pd.datetime.now(),' | ', 'Finished side-by-side build')
print(pd.datetime.now(),' | ', 'Side by side pairs sample:')
print(Xsbs.sample(5))

ix_con_singlecol = escon.multiindex21column(on_ix=ix_con_multi)


## Explore

from suricate.explore import Explorer

n_questions = 100
print(pd.datetime.now(),' | ', 'Starting questions with n_questions = {}'.format(n_questions))

# Loading the cheatsheet
y_true = rebuild_ytrue(ix= ix_con_multi)

## Fit the cluster to non-supervized data

print(pd.datetime.now(),' | ', 'Starting cluster fit with unsupervized data')
exp = Explorer(n_simple=n_questions, n_hard=n_questions)
exp.fit_cluster(X=Xst[['es_score']])
y_cluster = pd.Series(data=exp.pred_cluster(X=Xst), index=Xst.index, name='y_cluster')
X_cluster = pd.DataFrame(y_cluster)
print(pd.datetime.now(),' | ', 'Done')

### Ask simple questions
ix_simple = exp.ask_simple(X=Xst)
Sbs_simple = Xsbs.loc[ix_simple]
y_simple = y_true.loc[ix_simple]
print(pd.datetime.now(),' | ', 'Result of simple questions:')
print(Sbs_simple.sample(10))


### Fit the cluser with supervized data
print(pd.datetime.now(),' | ', 'Start fitting the cluster classifier with supervized data:')
exp.fit(X=Xst, y=y_simple, fit_cluster=False)
print(pd.datetime.now(),' | ', 'Done')

### Ask hard (pointed) questions
ix_hard = exp.ask_hard(X=Xst, y=y_simple)
Sbs_hard = Xsbs.loc[ix_hard]
y_hard = y_true.loc[ix_hard]
print(pd.datetime.now(),' | ', 'Result of hard questions:')
print(Sbs_hard.sample(10))

### Obtain the results of the labels
y_questions = y_true.loc[ix_hard.union(ix_simple)]
X_questions = Xsbs.loc[y_questions.index].copy()


### Start further matching

from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import FeatureUnion

print(pd.datetime.now(),' | ', 'Start Pruning')

pruning_threshold = 15

### Make the pruning step
ix_further = Xst.loc[Xst['es_score'] > pruning_threshold].index
Xst_further = Xst.loc[ix_further]
Xsbs_further = Xsbs.loc[ix_further]
y_true_further = y_true.loc[ix_further]
print(pd.datetime.now(),' | ', 'Pruning ratio: {}'.format(len(ix_further)/Xst.shape[0]))


print(pd.datetime.now(),' | ', 'Starting further scoring')

_sbs_score_list = [
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='fuzzy')),
    ('street_fuzzy', SbsApplyComparator(on='street', comparator='fuzzy')),
    ('name_token', SbsApplyComparator(on='name', comparator='token')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='fuzzy')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='fuzzy')),
    ('postalcode_contains', SbsApplyComparator(on='postalcode', comparator='contains'))
]

scorer_sbs = FeatureUnion(transformer_list=_sbs_score_list)
scores_further = scorer_sbs.fit_transform(X=Xsbs_further)
scores_further = pd.DataFrame(data=scores_further, index=ix_further, columns=[c[0] for c in _sbs_score_list])
scores_further = pd.concat([Xst_further, scores_further], axis=1, ignore_index=False)
print(pd.datetime.now(),' | ', 'Done')
print(pd.datetime.now(),' | ', 'Scores:')
print(scores_further.sample(10))

from suricate.pipeline import PartialClf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

### Make the pipeline
pipe = Pipeline(steps=[
    ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
    ('Scaler', Normalizer()),
    ('PCA', PCA(n_components=4)),
    ('Predictor', GradientBoostingClassifier(n_estimators=500))
])

print(pd.datetime.now(),' | ', 'Launch prediction pipeline fit')
pred = PartialClf(classifier=pipe)
pred.fit(X=scores_further, y=y_true_further)
print(pd.datetime.now(),' | ', 'Done')
print(pd.datetime.now(),' | ', 'Scores on training data')
print(pred.score(X=scores_further, y=y_true_further))
y_pred = pred.predict(X=scores_further)
print(pd.datetime.now(),' | ', 'Number of true and false values')
print(y_pred.value_counts())



print(pd.datetime.now(),' | ', 'Starting saving the results to SQL')
df_source.to_sql(name='df_source', con=engine, if_exists='replace', index=True)
df_target.to_sql(name='df_target', con=engine, if_exists='replace', index=True)
print(pd.datetime.now(),' | ', 'pushed to sql tables df_source and df_target')

Xst.reset_index(
    drop=False
).set_index(
    pd.Series(data=ix_con_singlecol, name='ix')
).to_sql(
    name='es_scores', con=engine, if_exists='replace'
)

Xsbs.reset_index(
    drop=False
).set_index(
    pd.Series(data=ix_con_singlecol, name='ix')
).to_sql(
    name='es_sbs', con=engine, if_exists='replace'
)

print(pd.datetime.now(), ' | ', 'Pushed Scores and Side-by-Side view to SQL with table names es_scores and es_sbs')

X_cluster['avg_score'] = Xst[['es_score']].mean(axis=1)

X_cluster['y_true'] = y_true
X_cluster['ix'] = ix_con_singlecol
X_cluster.reset_index(
    drop=False
).set_index(
    'ix'
)[
    ['ix_source', 'ix_target', 'avg_score', 'y_cluster', 'y_true']
].to_sql('cluster_output', con=engine, if_exists='replace')
print(pd.datetime.now(),' | ', 'Pushed X_cluster to SQL with name cluster_output')

X_questions.to_sql('questions', con=engine, if_exists='replace')
print(pd.datetime.now(),' | ', 'Pushed X_questions to SQL with name questions')

scores_further.to_sql('scores_final', con=engine, if_exists='replace')
print(pd.datetime.now(),' | ', 'Pushed scores_further to SQL with name scores_further')
