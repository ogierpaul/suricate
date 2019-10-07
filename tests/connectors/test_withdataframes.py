from sklearn.pipeline import make_union

from suricate.lrdftransformers import VectorizerConnector, ExactConnector
from suricate.data.base import ix_names
from suricate.data.companies import getleft, getright,  getXlr, getytrue
left = getleft(nrows=100)
right = getright(nrows=100)
X_lr = getXlr(nrows=100)
y_true = getytrue(Xlr=X_lr)

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
        ExactConnector(on='duns', pruning=False)

    ]
    scorer = make_union(*stages)
    X_score = scorer.transform(X=X_lr)
    assert X_score.shape[0] == expected_shape
    pass