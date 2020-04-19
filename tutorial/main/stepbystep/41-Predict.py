from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
import pandas as pd
from suricate.pipeline import PartialClf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier

engine = create_engine_ready()

Xscores = pd.read_sql(sql="SELECT * FROM scores_final", con=engine).set_index(
    ['ix_source', 'ix_target'], drop=True)
ix_double = Xscores.index
ix_single = Xscores['ix']
# Xtc = pd.read_sql(sql="SELECT * FROM es_scores", con=engine).set_index(['ix_source', 'ix_target'], drop=True)[['ix', 'es_score']]
# Xsbs = pd.read_sql(sql="SELECT * FROM es_sbs", con=engine).set_index(['ix_source', 'ix_target'], drop=True)


# REBUILD Y_true
y_true = pd.read_sql(sql="SELECT * FROM y_true WHERE y_true = 1", con=engine).set_index(['ix_source', 'ix_target'],
                                                                                        drop=True)
y_truetemp = pd.DataFrame(ix_single)
y_truetemp['y_true'] = 0
y_truetemp.loc[y_true.index.intersection(ix_double), 'y_true'] = y_true.loc[
    y_true.index.intersection(ix_double), 'y_true']
y_true = y_truetemp.copy()['y_true']
del y_truetemp
### y_true has now a multiindex, ix, and y_true columns


Xscores = Xscores[[c for c in Xscores.columns if c != 'ix']]
### Make the pipeline
pipe = Pipeline(steps=[
    ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
    ('Scaler', Normalizer()),
    ('PCA', PCA(n_components=4)),
    ('Predictor', GradientBoostingClassifier(n_estimators=500))
])
pred = PartialClf(classifier=pipe)
pred.fit(X=Xscores, y=y_true)
print(pred.score(X=Xscores, y=y_true))
y_pred = pred.predict(X=Xscores)
