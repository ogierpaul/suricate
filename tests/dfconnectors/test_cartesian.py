from unittest import TestCase

import pandas as pd

from wookie.connectors.dataframes import CartesianConnector
from wookie.connectors.dataframes.base import DFConnector
from wookie.connectors.dataframes.base import cartesian_join
from wookie.preutils import concatixnames

ixname = 'myindex'
lsuffix = 'left'
rsuffix = 'right'
ixnameleft, ixnameright, ixnamepairs = concatixnames(
    ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
)
samplecol = 'name'

left = pd.DataFrame(
    {
        ixname: [0, 1, 2],
        samplecol: ['foo', 'bar', 'ninja']
    }
).set_index(ixname)

right = pd.DataFrame(
    {
        ixname: [0, 1, 2],
        samplecol: ['foo', 'bar', 'baz']
    }
).set_index(ixname)

# The side by side should be with None
sidebyside = pd.DataFrame(
    [
        [0, "foo", 0, "foo", 1],  # case equal
        [0, "foo", 1, "bar", 0],  # case False
        [0, "foo", 2, "baz", 0],
        [1, "bar", 0, "foo", 0],
        [1, "bar", 1, "bar", 1],  # case True for two
        [1, "bar", 2, "baz", 1],  # case True for two fuzzy
        [2, "ninja", 0, "foo", 0],
        [2, "ninja", 0, "bar", 0],
        [2, "ninja", 0, "baz", 0]
        #       [2, "ninja", None]  # case None --> To be removed --> No y_true
    ],
    columns=[ixnameleft, '_'.join([samplecol, lsuffix]), ixnameright, '_'.join([samplecol, rsuffix]), 'y_true']
)
y_true = sidebyside['y_true']


class TestCartesian_join(TestCase):
    def test_cartesian_join(self, left=left, right=right):
        df = cartesian_join(left=left, right=right, lsuffix=lsuffix, rsuffix=rsuffix)
        # Output is a DataFrame
        assert isinstance(df, pd.DataFrame)
        # output number of rows is the multiplication of both rows
        assert df.shape[0] == left.shape[0] * right.shape[0]
        # output number of columns are left columns + right columns + 2 columns for each indexes
        assert df.shape[1] == 2 + left.shape[1] + right.shape[1]
        # every column of left and right, + the index, is found with a suffix in the output dataframe
        for oldname in left.reset_index(drop=False).columns:
            newname = '_'.join([oldname, lsuffix])
            assert newname in df.columns
        for oldname in right.reset_index(drop=False).columns:
            newname = '_'.join([oldname, rsuffix])
            assert newname in df.columns

        # assert sidebyside == df
        pass


class TestCartesian(TestCase):
    def test_init(self, left=left, right=right):
        cart = CartesianConnector(left=left, right=right, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        assert isinstance(cart, DFConnector)
        assert isinstance(cart, CartesianConnector)

    def test_fit(self):
        cart = CartesianConnector(left=left, right=right, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        cart.fit()
        assert True

    def test_transform(self):
        cart = CartesianConnector(left=left, right=right, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        cart.fit()
        score = cart.transform()
        assert isinstance(score, pd.Series)
        assert score.shape[0] == left.shape[0] * right.shape[0]
        # output number of columns are left columns + right columns + 2 columns for each indexes
        sbs = cart.sidebyside()
        assert isinstance(sbs, pd.DataFrame)
        assert sbs.shape[1] == left.shape[1] + right.shape[1] + 1
        # every column of left and right, + the index, is found with a suffix in the output dataframe
        for oldname in left.columns:
            newname = '_'.join([oldname, lsuffix])
            assert newname in sbs.columns
        for oldname in right.columns:
            newname = '_'.join([oldname, rsuffix])
            assert newname in sbs.columns
        assert True

    def test_score(self):
        cart = CartesianConnector(left=left, right=right, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        cart.fit()
        score = cart.pruning_score(y_true=y_true)
        print(score)
        assert isinstance(score, dict)
