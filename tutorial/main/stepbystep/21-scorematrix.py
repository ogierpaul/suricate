from tutorial.main.stepbystep.stepbysteputils.esconnector import getesconnector
from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
import pandas as pd

escon = getesconnector()
engine = create_engine_ready()

# nrows = 10
# df_left = pd.read_sql(sql="SELECT * FROM df_left LIMIT {}".format(nrows), con=engine)
df_left = pd.read_sql(sql="SELECT * FROM df_left", con=engine)
df_left.set_index('ix', drop=True, inplace=True)
Xtc = escon.fit_transform(X=df_left)
ix = Xtc.index
Xsbs = escon.getsbs(X=df_left, on_ix=ix)

ix_singlecol = escon.multiindex21column(on_ix=ix)
for x in Xtc, Xsbs:
    x['ix']=ix_singlecol
    x.set_index('ix',drop=False)
Xtc.to_sql(name='es_scores', con=engine, if_exists='replace')
Xsbs.to_sql(name='es_sbs', con=engine, if_exists='replace')
print(df_left.shape[0])
print(Xsbs.shape[0])
