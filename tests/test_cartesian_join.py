from unittest import TestCase

import pandas as pd


class TestCartesian_join(TestCase):

    def test_function(self):
        from wookie.connectors import cartesian_join
        df1 = pd.DataFrame({'name': ['foo', 'bath']})
        df2 = pd.DataFrame({'name': ['foo', 'bar', 'baz']})
        df = cartesian_join(left_df=df1, right_df=df2)
        assert df.shape[0] == df1.shape[0] * df2.shape[0]
        assert df.shape[1] == df1.shape[1] + df2.shape[1] + 2
        # check that the column names are correctly renamed, including index
        columns = set(df.columns.tolist())
        answer = {'name_left', 'name_right', 'index_left', 'index_right'}
        assert columns == answer
        # check that the count of values are counted right
        for c in df1.name.values:
            assert df['name_left'].value_counts().loc[c] == 3
        for c in df2.name.values:
            assert df['name_right'].value_counts().loc[c] == 2
