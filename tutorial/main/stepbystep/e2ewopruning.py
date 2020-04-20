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
df_source_raw = getsource(nrows=500)
df_target_raw = gettarget(nrows=None)


from sklearn.model_selection import train_test_split

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


import pandas as pd

from tutorial.main.stepbystep.stepbysteputils.esconnector import getesconnector

escon = getesconnector()


from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import FeatureUnion


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



left_train, left_test = train_test_split(df_source_raw, train_size=0.5)

Xst_train = escon.fit_transform(X=left_train)
ix_con_train = Xst_train.index
Xsbs_train = escon.getsbs(X=left_train, on_ix=ix_con_train)
scores_further_train = scorer_sbs.fit_transform(X=Xsbs_train)
scores_further_train = pd.DataFrame(data=scores_further_train, index=ix_con_train, columns=[c[0] for c in _sbs_score_list])
scores_further_train = pd.concat([Xst_train[['es_score']], scores_further_train], axis=1, ignore_index=False)
y_true_train = rebuild_ytrue(ix=ix_con_train)
pred.fit(X=scores_further_train, y=y_true_train)
y_pred_train = pred.predict(X=scores_further_train)

print(pd.datetime.now(),' | ', 'Scores on training data')
print(pd.datetime.now(),' | ', 'accuracy: {}'.format(accuracy_score(y_true=y_true_train, y_pred=y_pred_train)))
print(pd.datetime.now(),' | ', 'precision: {}'.format(precision_score(y_true=y_true_train, y_pred=y_pred_train)))
print(pd.datetime.now(),' | ', 'recall: {}'.format(recall_score(y_true=y_true_train, y_pred=y_pred_train)))


Xst_test = escon.transform(X=left_test)
ix_con_test = Xst_test.index
Xsbs_test = escon.getsbs(X=left_test, on_ix=ix_con_test)
scores_further_test = scorer_sbs.transform(X=Xsbs_test)
scores_further_test = pd.DataFrame(data=scores_further_test, index=ix_con_test, columns=[c[0] for c in _sbs_score_list])
scores_further_test = pd.concat([Xst_test[['es_score']], scores_further_test], axis=1, ignore_index=False)
y_true_test = rebuild_ytrue(ix=ix_con_test)
y_pred_test = pred.predict(X=scores_further_test)

print(pd.datetime.now(),' | ', 'Scores on testing data')
print(pd.datetime.now(),' | ', 'accuracy: {}'.format(accuracy_score(y_true=y_true_test, y_pred=y_pred_test)))
print(pd.datetime.now(),' | ', 'precision: {}'.format(precision_score(y_true=y_true_test, y_pred=y_pred_test)))
print(pd.datetime.now(),' | ', 'recall: {}'.format(recall_score(y_true=y_true_test, y_pred=y_pred_test)))




