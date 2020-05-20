import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV as Classifier
from sklearn.pipeline import make_union, make_pipeline


from suricate.dftransformers import VectorizerConnector, ExactConnector, DfVisualSbs
from suricate.pipeline import PipeDfClf, PipeSbsClf
from suricate.sbstransformers import SbsApplyComparator

from suricate.data.companies import getXst, getytrue


def test_lrmodel():
    X_lr = getXst(nrows=100)
    y_true = getytrue(Xst=X_lr)
    scorer = make_union(*[
        VectorizerConnector(on='name', analyzer='char'),
        VectorizerConnector(on='street', analyzer='char'),
        ExactConnector(on='countrycode'),
        ExactConnector(on='postalcode'),
        ExactConnector(on='duns')
    ])
    imp = SimpleImputer(strategy='constant', fill_value=0)
    transformer = make_pipeline(*[scorer, imp])
    clf = Classifier()
    mypipe = PipeDfClf(transformer=transformer, classifier=clf)
    X_score = mypipe.transformer.fit_transform(X=X_lr)
    mypipe.fit(X=X_lr, y=y_true)
    print(mypipe.score(X=X_lr, y=y_true))


def test_sbsmodel():
    X_lr = getXst(nrows=100)
    y_true = getytrue(Xst=X_lr)
    df_sbs = DfVisualSbs().fit_transform(X=X_lr)
    df_sbs = df_sbs.loc[y_true.index]
    transformer = make_union(*[
        SbsApplyComparator(on='name', comparator='simple'),
        SbsApplyComparator(on='name', comparator='token'),
        SbsApplyComparator(on='street', comparator='simple')
    ])
    imp = SimpleImputer(strategy='constant', fill_value=0)
    transformer = make_pipeline(*[transformer, imp])
    clf = Classifier()
    mypipe = PipeSbsClf(transformer=transformer, classifier=clf)
    mypipe.fit(X=df_sbs, y=y_true)
    print(mypipe.score(X=df_sbs, y=y_true))


