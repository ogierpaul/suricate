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
df_source_raw = getsource(nrows=None)
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
    ('name_fuzzy', SbsApplyComparator(on='name', comparator='fuzzy')),
    ('street_fuzzy', SbsApplyComparator(on='street', comparator='fuzzy')),
    ('name_token', SbsApplyComparator(on='name', comparator='token')),
    ('street_token', SbsApplyComparator(on='street', comparator='token')),
    ('city_fuzzy', SbsApplyComparator(on='city', comparator='fuzzy')),
    ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='fuzzy')),
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


Xst = escon.fit_transform(X=df_source)
ix_con = Xst.index
Xsbs = escon.getsbs(X=df_source, on_ix=ix_con)
scores_further = scorer_sbs.fit_transform(X=Xsbs)
scores_further = pd.DataFrame(data=scores_further, index=ix_con, columns=[c[0] for c in _sbs_score_list])
scores_further = pd.concat([Xst[['es_score']], scores_further], axis=1, ignore_index=False)
y_true = rebuild_ytrue(ix=ix_con)

X = scores_further
from sklearn.model_selection import cross_validate
scoring = ['precision', 'recall', 'accuracy']
print(pd.datetime.now(), ' | starting score')
pipe1 = Pipeline(steps=[
    ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
    ('Scaler', Normalizer()),
    ('PCA', PCA(n_components=4)),
    ('Predictor', GradientBoostingClassifier(n_estimators=1000, max_depth=5))
])

scores1 = cross_validate(estimator=pipe1, X=X, y=y_true, scoring=scoring, cv=5)
scores2 = cross_validate(estimator=pipe2, X=X, y=y_true, scoring=scoring, cv=5)
for c in scoring:
    print(pd.datetime.now(), ' | {} score1: {}'.format(c, np.average(scores1['test_'+c])))
    print(pd.datetime.now(), ' | {} score2: {}'.format(c, np.average(scores2['test_'+c])))




