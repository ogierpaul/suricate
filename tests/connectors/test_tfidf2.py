from sklearn.pipeline import make_union

from suricate.dftransformers.vectorizer import VectorizerConnector
from suricate.data.base import ix_names
from suricate.data.companies import getsource, gettarget, getXst, getytrue
left = getsource(nrows=100)
right = gettarget(nrows=100)
X_lr = getXst(nrows=100)
y_true = getytrue(Xst=X_lr)


def test_loaddata():
    print(ix_names['ixname'])
    print(left.shape[0])
    print(right.shape[0])
    assert True




def test_tfidf():
    expected_shape = left.shape[0] * right.shape[0]
    stages = [
        VectorizerConnector(on='name', analyzer='char', pruning=False),
        VectorizerConnector(on='street', analyzer='char', pruning=False),
    ]
    scorer = make_union(*stages)
    scorer.fit(X=X_lr)
    X_score = scorer.transform(X=X_lr)
    assert X_score.shape[0] == expected_shape
    pass