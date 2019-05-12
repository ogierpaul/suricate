import pandas as pd
from sklearn.linear_model import LogisticRegressionCV as Classifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

from wookie.lrdftransformers import VectorizerConnector, ExactConnector, FuzzyConnector
from ..data.bar import ix_names, df_X, df_sbs

y_train = 0


def test_pipeline(ix_names=ix_names, df_X=df_X, df_sbs=df_sbs):
    y_true = df_sbs['y_true']
    scorer = make_union(*[
        VectorizerConnector(on='name', analyzer='char'),
        ExactConnector(on='name'),
        FuzzyConnector(on='name')
    ])
    clf = Classifier()
    X_scores = pd.DataFrame(scorer.transform(X=df_X))
    X_scores.fillna(0, inplace=True)
    clf.fit(X=X_scores, y=y_true)
    y_pred = clf.predict(X=X_scores)
    print(accuracy_score(y_true=y_true, y_pred=y_pred))

    pass


def test_pipeline2(ix_names=ix_names, df_X=df_X, df_sbs=df_sbs):
    y_true = df_sbs['y_true']
    scorer = make_union(*[
        VectorizerConnector(on='name', analyzer='char'),
        ExactConnector(on='name'),
        FuzzyConnector(on='name')
    ])
    imp = Imputer()
    clf = Classifier()
    pipe = make_pipeline(*[
        scorer, imp, clf
    ])
    pipe.fit(X=df_X, y=y_true)
    y_pred = pipe.predict(X=df_X)
    print(accuracy_score(y_true=y_true, y_pred=y_pred))
    pass
