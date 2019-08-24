from suricate.lrdftransformers import LrDfVisualHelper
from suricate.data.companies import getXlr
import pandas as pd


class Test_LrDfVisualHelper():
    def check_output(self, X, cols=('name', 'street', 'city', 'postalcode', 'countrycode', 'duns'), lsuffix='left', rsuffix='right', ix='ix'):
        assert isinstance(X, pd.DataFrame)
        ixname = (ix + '_' + lsuffix, ix + '_' + rsuffix )
        assert set(X.index.names) == set(ixname)
        for c in cols:
            assert c + '_' + lsuffix in X.columns
            assert c + '_' + rsuffix in X.columns
        return True

    def test_fit(self):
        nrows = 100
        Xlr = getXlr(nrows=100)
        viz = LrDfVisualHelper()
        viz.fit(X=Xlr)
        Xsbs = viz.transform(X=Xlr)
        assert Xsbs.shape[0] == Xlr[0].shape[0]*Xlr[1].shape[0]
        assert Xsbs.shape[1] == Xlr[0].shape[1] + Xlr[1].shape[1]
        assert self.check_output(X=Xsbs)
        return True

    def test_fit_transform(self):
        nrows = 100
        Xlr = getXlr(nrows=100)
        viz = LrDfVisualHelper()
        Xsbs = viz.fit_transform(X=Xlr)
        assert Xsbs.shape[0] == Xlr[0].shape[0]*Xlr[1].shape[0]
        assert Xsbs.shape[1] == Xlr[0].shape[1] + Xlr[1].shape[1]
        assert self.check_output(X=Xsbs)
        return True
