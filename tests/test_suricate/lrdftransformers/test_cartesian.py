from suricate.dftransformers.cartesian import DfVisualSbs, cartesian_join
from suricate.data.companies import getXst
import pandas as pd
import pytest


class Test_LrDfVisualHelper():
    def check_output(self, X, cols=('name', 'street', 'city', 'postalcode', 'countrycode', 'duns'), source_suffix='source', target_suffix='target', ix='ix'):
        assert isinstance(X, pd.DataFrame)
        ixname = (ix + '_' + source_suffix, ix + '_' + target_suffix )
        assert set(X.index.names) == set(ixname)
        for c in cols:
            assert c + '_' + source_suffix in X.columns
            assert c + '_' + target_suffix in X.columns
        return True

    def test_fit(self):
        nrows = 100
        Xst = getXst(nrows=100)
        viz = DfVisualSbs()
        viz.fit(X=Xst)
        Xsbs = viz.transform(X=Xst)
        assert Xsbs.shape[0] == Xst[0].shape[0]*Xst[1].shape[0]
        assert Xsbs.shape[1] == Xst[0].shape[1] + Xst[1].shape[1]
        assert self.check_output(X=Xsbs)
        return True

    def test_fit_transform(self):
        nrows = 100
        Xst = getXst(nrows=100)
        viz = DfVisualSbs()
        Xsbs = viz.fit_transform(X=Xst)
        assert Xsbs.shape[0] == Xst[0].shape[0]*Xst[1].shape[0]
        assert Xsbs.shape[1] == Xst[0].shape[1] + Xst[1].shape[1]
        assert self.check_output(X=Xsbs)
        return True

class Test_cartesian_join():
    def test_normal_behaviour(self):
        df_source = pd.DataFrame({'ix': range(3), 'name':list('abc')}).set_index('ix')
        df_target = pd.DataFrame({'ix': range(3), 'name':list('abc')}).set_index('ix')
        on_ix = pd.MultiIndex.from_tuples([(0,0), (1,1), (2, 0)], names=['ix_source', 'ix_target'])
        Xsbs = cartesian_join(source=df_source, target=df_target, on_ix=on_ix)
        assert on_ix.shape[0] == len(on_ix)
        assert {'name_source', 'name_target'} == set(Xsbs.columns)

