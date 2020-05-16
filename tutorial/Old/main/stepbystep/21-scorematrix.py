from tutorial.Old.main.stepbystep.stepbysteputils.esconnector import getesconnector
from tutorial.Old.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
import pandas as pd

escon = getesconnector()
engine = create_engine_ready()

# nrows = 10
# df_source = pd.read_sql(sql="SELECT * FROM df_source LIMIT {}".format(nrows), con=engine)
df_source = pd.read_sql(sql="SELECT * FROM df_source", con=engine)
df_source.set_index('ix', drop=True, inplace=True)
Xst = escon.fit_transform(X=df_source)
ix = Xst.index
Xsbs = escon.getsbs(X=df_source, on_ix=ix)

ix_singlecol = escon.multiindex21column(on_ix=ix)
for x in Xst, Xsbs:
    x['ix']=ix_singlecol
    x.set_index('ix',drop=False)
Xst.to_sql(name='es_scores', con=engine, if_exists='replace')
Xsbs.to_sql(name='es_sbs', con=engine, if_exists='replace')
print(df_source.shape[0])
print(Xsbs.shape[0])
