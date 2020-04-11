# End to End Dedupe
## Pruning pipeline using Elastic search

from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
from suricate.data.companies import getleft, getright
import pandas as pd
import  numpy as np

engine = create_engine_ready()

# filefolder = '~/'
# leftpath = 'left.csv'
# rightpath = 'right.csv'
# df_left = pd.read_csv(filefolder + leftpath, index_col=0, sep='|', encoding='utf-8')
# df_right = pd.read_csv(filefolder + rightpath, index_col=0, sep='|', encoding='utf-8')
df_left_raw = getleft(nrows=100)
df_right_raw = getright(nrows=100)

def rebuild_ytrue(ix):
    y_true_saved = pd.read_sql(sql="SELECT * FROM y_true WHERE y_true.y_true = 1", con=engine).set_index(
        ['ix_left', 'ix_right'],
        drop=True)['y_true']
    y = pd.Series(index=ix, data = np.zeros(shape=len(ix)), name='y_true')
    ix_common = y_true_saved.index.intersection(ix)
    y.loc[ix_common] = y_true_saved.loc[ix_common]
    return y



def prepare_left(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2


def prepare_right(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2


df_left = prepare_left(df_left_raw)
df_right = prepare_right(df_right_raw)
assert df_left.columns.equals(df_right.columns)
print(pd.datetime.now(),' | ', 'number of rows on left:{}'.format(df_left.shape[0]))
print(pd.datetime.now(),' | ', 'number of rows on right:{}'.format(df_right.shape[0]))


## Push the data to ES

import elasticsearch
import pandas as pd
from suricate.dbconnectors.esconnector import index_with_es
import time


## Put the data to ES, drop the index first and then re create
esclient = elasticsearch.Elasticsearch()
es_indice = 'df_right'

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
    index_with_es(client=esclient, df=df_right, index=es_indice, ixname="ix", reset_index=True, doc_type='_doc')
    time.sleep(5)
pass
catcount = esclient.count(index=es_indice)['count']
assert catcount == df_right.shape[0]
print(pd.datetime.now(),' | ', 'pushed to es_sql indice {}'.format(es_indice))
print(pd.datetime.now(),' | ', 'number of docs: {}'.format(catcount))


## Connect the data
from tutorial.main.stepbystep.stepbysteputils.esconnector import getesconnector

print(pd.datetime.now(),' | ', 'Starting connection')
escon = getesconnector()

Xtc = escon.fit_transform(X=df_left)
print(pd.datetime.now(),' | ', 'Finished connection')
print(pd.datetime.now(),' | ', 'number of pairs {}'.format(Xtc.shape[0]))
print(pd.datetime.now(),' | ', 'Connection scores sample:')
print(Xtc.sample(5))

ix_con_multi = Xtc.index
print(pd.datetime.now(),' | ', 'Starting side-by-side build')
Xsbs = escon.getsbs(X=df_left, on_ix=ix_con_multi)
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
exp.fit_cluster(X=Xtc[['es_score']])
y_cluster = pd.Series(data=exp.pred_cluster(X=Xtc), index=Xtc.index, name='y_cluster')
X_cluster = pd.DataFrame(y_cluster)
print(pd.datetime.now(),' | ', 'Done')

### Ask simple questions
ix_simple = exp.ask_simple(X=Xtc)
Sbs_simple = Xsbs.loc[ix_simple]
y_simple = y_true.loc[ix_simple]
print(pd.datetime.now(),' | ', 'Result of simple questions:')
print(Sbs_simple.sample(10))


### Fit the cluser with supervized data
print(pd.datetime.now(),' | ', 'Start fitting the cluster classifier with supervized data:')
exp.fit(X=Xtc, y=y_simple, fit_cluster=False)
print(pd.datetime.now(),' | ', 'Done')

### Ask hard (pointed) questions
ix_hard = exp.ask_hard(X=Xtc, y=y_simple)
Sbs_hard = Xsbs.loc[ix_hard]
y_hard = y_true.loc[ix_hard]
print(pd.datetime.now(),' | ', 'Result of hard questions:')
print(Sbs_hard.sample(10))

### Obtain the results of the labels
y_questions = y_true.loc[ix_hard.union(ix_simple)]
X_questions = Xsbs.loc[y_questions.index].copy()


### Start further matching

from suricate.sbsdftransformers import FuncSbsComparator
from sklearn.pipeline import FeatureUnion

print(pd.datetime.now(),' | ', 'Start Pruning')

pruning_threshold = 15

### Make the pruning step
ix_further = Xtc.loc[Xtc['es_score'] > pruning_threshold].index
Xtc_further = Xtc.loc[ix_further]
Xsbs_further = Xsbs.loc[ix_further]
y_true_further = y_true.loc[ix_further]
print(pd.datetime.now(),' | ', 'Pruning ratio: {}'.format(len(ix_further)/Xtc.shape[0]))


print(pd.datetime.now(),' | ', 'Starting further scoring')

_sbs_score_list = [
    ('name_fuzzy', FuncSbsComparator(on='name', comparator='fuzzy')),
    ('street_fuzzy', FuncSbsComparator(on='street', comparator='fuzzy')),
    ('name_token', FuncSbsComparator(on='name', comparator='token')),
    ('street_token', FuncSbsComparator(on='street', comparator='token')),
    ('city_fuzzy', FuncSbsComparator(on='city', comparator='fuzzy')),
    ('postalcode_fuzzy', FuncSbsComparator(on='postalcode', comparator='fuzzy')),
    ('postalcode_contains', FuncSbsComparator(on='postalcode', comparator='contains'))
]

scorer_sbs = FeatureUnion(transformer_list=_sbs_score_list)
scores_further = scorer_sbs.fit_transform(X=Xsbs_further)
scores_further = pd.DataFrame(data=scores_further, index=ix_further, columns=[c[0] for c in _sbs_score_list])
scores_further = pd.concat([Xtc_further, scores_further], axis=1, ignore_index=False)
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
df_left.to_sql(name='df_left', con=engine, if_exists='replace', index=True)
df_right.to_sql(name='df_right', con=engine, if_exists='replace', index=True)
print(pd.datetime.now(),' | ', 'pushed to sql tables df_left and df_right')

Xtc.reset_index(
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

X_cluster['avg_score'] = Xtc[['es_score']].mean(axis=1)

X_cluster['y_true'] = y_true
X_cluster['ix'] = ix_con_singlecol
X_cluster.reset_index(
    drop=False
).set_index(
    'ix'
)[
    ['ix_left', 'ix_right', 'avg_score', 'y_cluster', 'y_true']
].to_sql('cluster_output', con=engine, if_exists='replace')
print(pd.datetime.now(),' | ', 'Pushed X_cluster to SQL with name cluster_output')

X_questions.to_sql('questions', con=engine, if_exists='replace')
print(pd.datetime.now(),' | ', 'Pushed X_questions to SQL with name questions')

scores_further.to_sql('scores_final', con=engine, if_exists='replace')
print(pd.datetime.now(),' | ', 'Pushed scores_further to SQL with name scores_further')
