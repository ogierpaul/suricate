import pandas as pd
from sklearn.linear_model import LogisticRegressionCV as Classifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

from wookie.pandasconnectors import VectorizerConnector, ExactConnector, FuzzyConnector
from wookie.pipeline import LrModel
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


# THIS LAST TEST FAILED AND I NEED TO REBUILD A NEW METHOD
# I HAVE A K.O because the output of a pipeline is a np.ndarray
# but my indexer has an index
# I cannot a posteriori link the nump.ndarray (X_score) with the y_true index
def test_pipeline3(ix_names=ix_names, df_X=df_X, df_sbs=df_sbs):
    y_true = df_sbs['y_true'].sample(7)
    scorer = make_union(*[
        VectorizerConnector(on='name', analyzer='char'),
        ExactConnector(on='name'),
        FuzzyConnector(on='name')
    ])
    imp = Imputer()
    scorer = make_pipeline(*[scorer, imp])
    clf = Classifier()
    mypipe = LrModel(scorer=scorer, classifier=clf)
    mypipe.fit(X=df_X, y=y_true)
    y_pred = mypipe.predict(X=df_X)
    print(mypipe.score(X=df_X, y=y_true))
    # print(accuracy_score(y_true=y_true, y_pred=y_pred))
    pass
