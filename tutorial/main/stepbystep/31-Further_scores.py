from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
import pandas as pd
from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import FeatureUnion

engine = create_engine_ready()

pruning_threshold = 15


# nrows = 50
# Xst = pd.read_sql(sql="SELECT * FROM es_scores LIMIT {}".format(nrows), con=engine).set_index(['ix_source', 'ix_target'], drop=True)[['ix', 'es_score']]
# Xsbs = pd.read_sql(sql="SELECT * FROM es_sbs LIMIT {}".format(nrows), con=engine).set_index(['ix_source', 'ix_target'], drop=True)
Xst = pd.read_sql(sql="SELECT * FROM es_scores", con=engine).set_index(['ix_source', 'ix_target'], drop=True)[['ix', 'es_score']]
Xsbs = pd.read_sql(sql="SELECT * FROM es_sbs", con=engine).set_index(['ix_source', 'ix_target'], drop=True)


# REBUILD Y_true
y_true = pd.read_sql(sql="SELECT * FROM y_true WHERE y_true.y_true = 1", con=engine).set_index(['ix_source', 'ix_target'], drop=True)
y_truetemp=Xst[['ix']]
y_truetemp['y_true']=0
y_truetemp.loc[y_true.index.intersection(Xst.index), 'y_true'] = y_true.loc[y_true.index.intersection(Xst.index), 'y_true']
y_true = y_truetemp.copy()
del y_truetemp
### y_true has now a multiindex, ix, and y_true columns

### Make the pruning step
ix_further = Xst.loc[Xst['es_score'] > pruning_threshold].index
Xst = Xst.loc[ix_further]
Xsbs = Xsbs.loc[ix_further]
y_true = y_true.loc[ix_further]

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
scores = scorer_sbs.fit_transform(X=Xsbs)
scores = pd.DataFrame(data=scores, index=ix_further, columns=[c[0] for c in _sbs_score_list])
for c in ['ix', 'es_score']:
    scores[c] = Xst[c]
scores.reset_index(inplace=True, drop=False)
scores.set_index('ix', inplace = True)
scores.to_sql('scores_final', con=engine, if_exists='replace')
