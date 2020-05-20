from suricate.sbstransformers import SbsApplyComparator
from suricate.dftransformers import DfApplyComparator
from suricate.data.circus import getXsbs, getXst

# from ..data.circus import circus_sbs, X_lr


def test_cclr():
    Xst = getXst()
    Xsbs = getXsbs()
    expected_shape = Xst[0].shape[0] * Xst[1].shape[0]
    assert Xsbs.shape[0] == expected_shape
    print(Xsbs)
    comp = SbsApplyComparator(on='name', comparator='simple')
    X_score = comp.transform(Xsbs)
    assert X_score.shape[0] == expected_shape
    Xsbs['score'] = X_score
    print(Xsbs.sort_values(by='score', ascending=False))

def test_sbs_tokensimple():
    Xsbs = getXsbs()
    ## Check that simple and token scores are different
    Xsbs['ratio'] =  SbsApplyComparator(on='name', comparator='simple').fit_transform(X=Xsbs)
    Xsbs['token'] =  SbsApplyComparator(on='name', comparator='token').fit_transform(X=Xsbs)
    Xsbs['diff'] = Xsbs['ratio'] - Xsbs['token']
    assert Xsbs['diff'].abs().sum() > 0
    print(Xsbs.loc[Xsbs['diff'].argmax()])

def test_df_tokensimple():
    ## Check that simple and token scores are different
    Xst = getXst()
    Xsbs = getXsbs()
    y_simple = DfApplyComparator(on='name', comparator='simple').fit_transform(X=Xst)
    y_token = DfApplyComparator(on='name', comparator='token').fit_transform(X=Xst)
    Xsbs['ratio'] = y_simple
    Xsbs['token'] = y_token
    Xsbs['diff'] = Xsbs['ratio'] - Xsbs['token']
    assert Xsbs['diff'].abs().sum() > 0
    print(Xsbs.loc[Xsbs['diff'].argmax()])