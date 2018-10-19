from unittest import TestCase

import pandas as pd

from wookie.connectors import cartesian_join, separatesides

ixname = 'myindex'
lsuffix = 'left'
rsuffix = 'right'
ixnameleft = ixname + '_' + lsuffix
ixnameright = ixname + '_' + rsuffix
ixnamepairs = [ixnameleft, ixnameright]
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
        [1, "bar", 1, "bar", 1],  # case True for two
        [1, "bar", 2, "baz", 1],  # case True for two fuzzy
        [2, "ninja", None]  # case None --> To be removed --> No y_true
    ],
    columns=[ixnameleft, '_'.join([samplecol, lsuffix]), ixnameright, '_'.join([samplecol, rsuffix]), 'y_true']
)


class TestCartesian_join(TestCase):
    def test_cartesian_join(self):
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


class TestSeparatesides(TestCase):
    def test_separatesides(self, sbs=sidebyside, lsuffix=lsuffix, rsuffix=rsuffix, y_true_col='y_true', ixname=ixname):
        assert isinstance(sbs, pd.DataFrame)
        newleft, newright, y_true = separatesides(
            df=sbs, lsuffix=lsuffix, rsuffix=rsuffix, y_true_col=y_true_col, ixname=ixname
        )
        self.newleft = newleft
        self.newright = newright
        self.y_true = y_true
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixname = ixname
        self.ixnameleft = self.ixname + '_' + self.lsuffix
        self.ixnameright = self.ixname + '_' + self.rsuffix

        # Check all the cols in the right place & they have the expected number of rows
        for sidedf, suffix in zip([self.newleft, self.newright], [self.lsuffix, self.rsuffix]):
            oldcols = [c for c in sidebyside.columns if c[-len(suffix):] == suffix]

            for oldcol in oldcols:
                newcol = oldcol[:-(len(suffix) + 1)]
                assert newcol in sidedf.reset_index(drop=False).columns
            tempixname = self.ixname + '_' + suffix
            assert sbs.drop_duplicates(subset=[tempixname]).shape[0] == sidedf.shape[0]
            assert pd.Index(sbs[tempixname].drop_duplicates()).difference(sidedf.index).shape[0] == 0

        # check ix_pairs:
        assert isinstance(self.y_true, pd.Series)
        assert set(y_true.index.names) == {self.ixnameleft, self.ixnameright}
        pass


def test_createsbs():
    pass
