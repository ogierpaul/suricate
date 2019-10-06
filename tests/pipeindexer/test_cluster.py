import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV as Classifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import MinMaxScaler

from suricate.data.companies import getXlr, getytrue
from suricate.lrdftransformers import VectorizerConnector, ExactConnector
from suricate.explore.clusterclassifier import ClusterClassifier
from suricate.pipeline import PipeSbsClf, PruningLrSbsClf, PipeLrClf
from suricate.preutils import createmultiindex
from suricate.preutils.metrics import scores
from suricate.sbsdftransformers import FuncSbsComparator


def test_pipelrcluster():
    X_lr = getXlr(nrows=100)
    y_true = getytrue(nrows=100)
    y_true = y_true.loc[
        y_true.index.intersection(createmultiindex(X=X_lr))
    ]
    scorer = make_union(*[
        VectorizerConnector(on='name', analyzer='word', ngram_range=(1, 2)),
        VectorizerConnector(on='street', analyzer='word', ngram_range=(1, 2)),
        ExactConnector(on='duns')
    ])
    imp = SimpleImputer(strategy='constant', fill_value=0)
    pca = PCA(n_components=2)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    cluster = KMeans(n_clusters=8)
    t2d = make_pipeline(*[scorer, imp, pca, scaler, cluster])
    clf1 = ClusterClassifier(cluster=cluster)
    lrmodel = PipeLrClf(transformer=t2d, classifier=clf1)
    y_pred_lr = lrmodel.fit_predict(X=X_lr, y=y_true)
    assert isinstance(y_pred_lr, np.ndarray)

    transformer2 = make_union(*[
        FuncSbsComparator(on='name', comparator='fuzzy'),
        FuncSbsComparator(on='name', comparator='token'),
        FuncSbsComparator(on='street', comparator='fuzzy'),
        FuncSbsComparator(on='city', comparator='fuzzy'),
        FuncSbsComparator(on='postalcode', comparator='fuzzy'),

    ])
    imp2 = SimpleImputer(strategy='constant', fill_value=0)
    transformer2 = make_pipeline(*[transformer2, imp2])
    clf2 = Classifier()
    sbsmodel = PipeSbsClf(transformer=transformer2, classifier=clf2)
    pipe = PruningLrSbsClf(lrmodel=lrmodel, sbsmodel=sbsmodel)
    pipe.fit(X=X_lr, y_lr=y_true, y_sbs=y_true)
    y_pred = pd.Series(
        data=pipe.predict(X=X_lr),
        index=createmultiindex(X=X_lr),
        name='y_pred'
    )
    y_pruning = pd.Series(data=np.where(y_pred_lr > 1, 1, y_pred_lr), index=createmultiindex(X=X_lr))
    pruningscores = scores(y_true=y_true, y_pred=y_pruning)
    finalscores = scores(y_true=y_true, y_pred=y_pred)
    print(pd.Series(y_pred.value_counts()))
    for c in pruningscores.keys():
        print('\n {}\n pruning:{}\nfinal{}\n'.format(c, pruningscores[c], finalscores[c]))
    assert True
