import pandas as pd
from sklearn.pipeline import make_union

from suricate.dftransformers import VectorizerConnector, DfApplyComparator
from suricate.data.base import ix_names
from suricate.data.circus import getXst, getytrue
from suricate.preutils.indextools import createmultiindex
from suricate.data.foo import getsource, gettarget, getXsbs, getXst, getytrue
left = getsource()
right = gettarget()
X_lr = getXst()
X_sbs = getXsbs()
y_true = getytrue()


def test_makeunion_y_true():
    X = X_lr
    stages = [
        VectorizerConnector(on='name', analyzer='char'),
        VectorizerConnector(on='name', analyzer='word'),
        DfApplyComparator(on='name', compfunc='simple'),
        DfApplyComparator(on='name', compfunc='token')
    ]
    pipe = make_union(*stages)
    pipe.fit(X=X)
    y = y_true
    pipe.fit(X=X, y=y)
    scores = pipe.transform(X=X)
    alldata = pd.DataFrame(scores, index=stages[0]._getindex(X=X), columns=[c.outcol for c in stages])
    print(alldata.columns)
    print(alldata)
    return None

