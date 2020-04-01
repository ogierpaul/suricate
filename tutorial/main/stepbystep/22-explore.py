from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
import pandas as pd
from suricate.explore import Explorer

engine = create_engine_ready()

nrows = 10
Xtc = pd.read_sql(sql="SELECT * FROM es_scores LIMIT {}".format(nrows), con=engine).set_index('ix', drop=True)
Xsbs = pd.read_sql(sql="SELECT * FROM es_sbs LIMIT {}".format(nrows), con=engine).set_index('ix', drop=True)
exp = Explorer(n_simple=50, n_hard=50)

