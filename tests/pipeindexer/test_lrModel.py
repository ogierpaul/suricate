import pandas as pd
from sklearn.linear_model import LogisticRegressionCV as Classifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

from wookie.lrdftransformers import VectorizerConnector, ExactConnector
from wookie.pipeline import PipeLrClf
from wookie.preutils import separatesides

# 'a7a163a4-eb5e-45a2-acde-597d99227959'

filepath_training = '/Users/paulogier/81-GithubPackages/wookie/operations/data/trainingdata.csv'
df_train = pd.read_csv(filepath_training, dtype=str, index_col=[0, 1]).sample(2000)
left, right, y = separatesides(df_train)
X = [left, right]


def test_lrmodel():
    transformer = make_union(*[
        VectorizerConnector(on='name', analyzer='char'),
        VectorizerConnector(on='street', analyzer='char'),
        ExactConnector(on='countrycode'),
        ExactConnector(on='postalcode'),
        ExactConnector(on='duns')
    ])
    imp = Imputer()
    transformer = make_pipeline(*[transformer, imp])
    clf = Classifier()
    mypipe = PipeLrClf(transformer=transformer, classifier=clf)
    mypipe.fit(X=X, y=y)
    print(mypipe.score(X=X, y=y))
