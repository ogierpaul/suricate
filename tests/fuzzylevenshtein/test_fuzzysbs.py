from suricate.sbsdftransformers import FuncSbsComparator
from suricate.data.circus import getXsbs, getXst

# from ..data.circus import circus_sbs, X_lr
X_lr = getXst()
X_sbs = getXsbs()

def test_cclr():
    expected_shape = X_lr[0].shape[0] * X_lr[1].shape[0]
    assert X_sbs.shape[0] == expected_shape
    print(X_sbs)
    comp = FuncSbsComparator(on='name', comparator='fuzzy')
    X_score = comp.transform(X_sbs)
    assert X_score.shape[0] == expected_shape
    X_sbs['score'] = X_score
    print(X_sbs.sort_values(by='score', ascending=False))
