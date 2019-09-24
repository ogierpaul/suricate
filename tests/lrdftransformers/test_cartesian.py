from suricate.lrdftransformers.cartesian import LrDfVisualHelper, create_lrdf_sbs
from suricate.data.companies import getXlr
import pandas as pd
import pytest


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

class Test_create_lrdf_sbs():
    def test_normal_behaviour(self):
        df_left = pd.DataFrame({'ix': range(3), 'name':list('abc')}).set_index('ix')
        df_right = pd.DataFrame({'ix': range(3), 'name':list('abc')}).set_index('ix')
        on_ix = pd.MultiIndex.from_tuples([(0,0), (1,1), (2, 0)], names=['ix_left', 'ix_right'])
        Xsbs = create_lrdf_sbs(X=[df_left, df_right], on_ix=on_ix)
        assert on_ix.shape[0] == len(on_ix)
        assert {'name_left', 'name_right'} == set(Xsbs.columns)

