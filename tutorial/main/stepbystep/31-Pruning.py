from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
import pandas as pd
from suricate.explore import Explorer
from suricate.base import multiindex21column

engine = create_engine_ready()

n_questions = 100

nrows = 50
Xtc = pd.read_sql(sql="SELECT * FROM es_scores LIMIT {}".format(nrows), con=engine).set_index(['ix_left', 'ix_right'], drop=True)[['ix', 'es_score']]
Xsbs = pd.read_sql(sql="SELECT * FROM es_sbs LIMIT {}".format(nrows), con=engine).set_index(['ix_left', 'ix_right'], drop=True)
# Xtc = pd.read_sql(sql="SELECT * FROM es_scores", con=engine).set_index(['ix_left', 'ix_right'], drop=True)[['ix', 'es_score']]
# Xsbs = pd.read_sql(sql="SELECT * FROM es_sbs", con=engine).set_index(['ix_left', 'ix_right'], drop=True)


# REBUILD Y_true
y_true = pd.read_sql(sql="SELECT * FROM y_true WHERE y_true.y_true = 1", con=engine).set_index(['ix_left', 'ix_right'], drop=True)
y_truetemp=Xtc[['ix']]
y_truetemp['y_true']=0
y_truetemp.loc[y_true.index.intersection(Xtc.index), 'y_true'] = y_true.loc[y_true.index.intersection(Xtc.index), 'y_true']
y_true = y_truetemp.copy()
del y_truetemp
### y_true has now a multiindex, ix, and y_true columns

### Make the pruning step
ix_further = Xtc.lox[Xtc['avg_score'] > 20].index
Xtc = Xtc.loc[ix_further]
Xsbs = Xsbs.loc[ix_further]

