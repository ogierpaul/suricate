import pandas as pd
from sklearn.pipeline import make_union

from suricate.lrdftransformers import VectorizerConnector, FuzzyConnector
from suricate.data.foo import ix_names
from suricate.data.circus import df_circus

def test_makeunionperso(ix_names, df_circus):
    X = df_circus
    stages = [
        VectorizerConnector(on='name', analyzer='char'),
        VectorizerConnector(on='name', analyzer='word'),
        FuzzyConnector(on='name', ratio='simple'),
        FuzzyConnector(on='name', ratio='token')
    ]
    pipe = make_union(*stages)
    pipe.fit(X=X)
    ix = pd.MultiIndex.from_product([[0], ['a', 'b', 'c']], names=ix_names['ixnamepairs'])
    y = pd.Series(index=ix, data=[1, 1, 0])
    pipe.fit(X=X, y=y)
    scores = pipe.transform(X=X)
    alldata = pd.DataFrame(scores, index=stages[0]._getindex(X=X), columns=[c.outcol for c in stages])
    print(alldata.columns)
    print(alldata)
    return None


def test_makeunion_y_true(ix_names, df_circus):
    X = df_circus
    stages = [
        VectorizerConnector(on='name', analyzer='char'),
        VectorizerConnector(on='name', analyzer='word'),
        FuzzyConnector(on='name', ratio='simple'),
        FuzzyConnector(on='name', ratio='token')
    ]
    pipe = make_union(*stages)
    pipe.fit(X=X)
    ix = pd.MultiIndex.from_product([[0], ['a', 'b', 'c']], names=ix_names['ixnamepairs'])
    y = pd.Series(index=ix, data=[1, 1, 0])
    pipe.fit(X=X, y=y)
    scores = pipe.transform(X=X)
    alldata = pd.DataFrame(scores, index=stages[0]._getindex(X=X), columns=[c.outcol for c in stages])
    print(alldata.columns)
    print(alldata)
    return None