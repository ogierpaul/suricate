import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV as Classifier
from sklearn.pipeline import make_union, make_pipeline

from suricate.preutils.functionclassifier import FunctionClassifier
from suricate.dftransformers import VectorizerConnector, ExactConnector, CartesianDataPasser, DfVisualHelper
from suricate.pipeline import PipeDfClf, PipeSbsClf, PruningDfSbsClf
from suricate.sbsdftransformers import FuncSbsComparator

from suricate.data.companies import getXst, getytrue

# from ..data.dataframes import X_lr, y_true, df_source, df_target

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
    df_sbs = DfVisualHelper().fit_transform(X=X_lr)
    df_sbs = df_sbs.loc[y_true.index]
    transformer = make_union(*[
        FuncSbsComparator(on='name', comparator='fuzzy'),
        FuncSbsComparator(on='name', comparator='token'),
        FuncSbsComparator(on='street', comparator='fuzzy')
    ])
    imp = SimpleImputer(strategy='constant', fill_value=0)
    transformer = make_pipeline(*[transformer, imp])
    clf = Classifier()
    mypipe = PipeSbsClf(transformer=transformer, classifier=clf)
    mypipe.fit(X=df_sbs, y=y_true)
    print(mypipe.score(X=df_sbs, y=y_true))


def test_pipeModel():
    X_lr = getXst(nrows=100)
    y_true = getytrue(Xst=X_lr)
    transformer1 = make_union(*[
        VectorizerConnector(on='name', analyzer='word'),
        VectorizerConnector(on='street', analyzer='word'),
        ExactConnector(on='countrycode'),
        ExactConnector(on='duns')
    ])
    imp1 = SimpleImputer(strategy='constant', fill_value=0)
    transformer1 = make_pipeline(*[transformer1, imp1])

    def myfunc(X):
        y_name = X[:, 0]
        y_street = X[:, 1]
        y_country = X[:, 2]
        y_duns = X[:, 3]
        y_return = np.logical_or(
            y_duns == 1,
            np.logical_and(
                y_country == 1,
                np.logical_or(
                    y_name > 0.3,
                    y_street > 0.3
                )
            )
        )
        return y_return

    clf1 = FunctionClassifier(func=myfunc)
    lrmodel = PipeDfClf(
        transformer=transformer1,
        classifier=clf1
    )
    transformer2 = make_union(*[
        FuncSbsComparator(on='name', comparator='fuzzy'),
        FuncSbsComparator(on='name', comparator='token'),
        FuncSbsComparator(on='street', comparator='fuzzy'),
        FuncSbsComparator(on='city', comparator='fuzzy'),
        FuncSbsComparator(on='postalcode', comparator='fuzzy'),

    ])
    imp2 = SimpleImputer(strategy='constant', fill_value=0)
    transformer2 = make_pipeline(*[transformer2, imp2])
    clf = Classifier()
    sbsmodel = PipeSbsClf(transformer=transformer2, classifier=clf)
    totalpipe = PruningDfSbsClf(lrmodel=lrmodel, sbsmodel=sbsmodel)
    totalpipe.fit(X=X_lr, y_lr=y_true, y_sbs=y_true)
    print(totalpipe.score(X=X_lr, y=y_true))
