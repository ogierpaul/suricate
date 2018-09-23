import pandas as pd
from sklearn.base import TransformerMixin

from wookie.comparators import exact_score


def cartesian_join(left_df, right_df, left_suffix='_left', right_suffix='_right'):
    """

    Args:
        left_df (pd.DataFrame): table 1
        right_df (pd.DataFrame): table 2
        left_suffix (str):
        right_suffix (str):

    Returns:
        pd.DataFrame

    Examples:
        df1 = pd.DataFrame({'a':['foo', 'bar']})
             a
        0   foo
        1	bar

        df2 = pd.DataFrame({'b':['foz', 'baz']})
             b
        0   foz
        1	baz

        cartesian_join(df1, df2)
            index_left	a_left	index_right	b_right
        0	0	        foo	    0	        foz
        1	0	        foo	    1	        baz
        2	1	        bar	    0	        foz
        3	1	        bar	    1	        baz

    """

    def rename_with_suffix(df, suffix):
        """
        rename the columns with a suffix, including the index
        Args:
            df (pd.DataFrame):
            suffix (str):

        Returns:
            pd.DataFrame

        Examples:
            df = pd.DataFrame({'a':['foo', 'bar']})
                 a
            0   foo
            1	bar

            rename_with_suffix(df, '_right')

                index_right	a_right
            0	0	        foo
            1	1	        bar
        """
        if suffix is None:
            return df
        assert isinstance(suffix, str)
        assert isinstance(df, pd.DataFrame)
        df_new = df.copy()
        df_new.index.name = 'index'
        df_new.reset_index(drop=False, inplace=True)
        cols = df_new.columns
        mydict = dict(
            zip(
                cols,
                map(lambda c: c + suffix, cols)
            )
        )
        df_new.rename(columns=mydict, inplace=True)
        return df_new

    # hack to create a column name unknown to both df1 and df2
    tempcolname = 'f1b3'
    while tempcolname in left_df.columns or tempcolname in right_df.columns:
        tempcolname += 'f'

    # create a new df1 with renamed cols
    df1new = rename_with_suffix(left_df, left_suffix)
    df2new = rename_with_suffix(right_df, right_suffix)
    df1new[tempcolname] = 0
    df2new[tempcolname] = 0
    dfnew = pd.merge(df1new, df2new, on=tempcolname).drop([tempcolname], axis=1)
    del df1new, df2new, tempcolname

    return dfnew


class BaseConnector(TransformerMixin):
    """
    Class inherited from TransformerMixin
    Attributes:
        attributesCols (list): list of column names desribing the data
        relevanceCol (str): name of column describing how relevant the data is to the query
        left_index (str): suffix to identify columns from the left dataset
        right_index (str): suffix to identify columns from the right dataset
    """

    def __init__(self, *args, **kwargs):
        TransformerMixin.__init__(self)
        self.attributesCols = []
        self.relevanceCol = []
        self.left_index = 'left_index'
        self.right_index = 'right_index'
        pass

    def transform(self, X, *_):
        """

        Args:
            X (pd.DataFrame): array containing the query
            *_:

        Returns:
            pd.DataFrame
        """
        assert isinstance(X, pd.DataFrame)
        X['relevance_score'] = None
        result = X
        return result

    def fit(self, *_):
        return self


class Cartesian(BaseConnector):
    """
    create a connector showing a cartesian join
    Attributes:
        reference (pd.DataFrame): reference data
        attributesCols (list):
        relevancesCols (list): ['relevance_score']
        relevance_func : callable
        relevance_threshold (float)
    Examples:
        from wookie.connectors import Cartesian
        df1 = pd.DataFrame({'name': ['foo', 'bath']})
        df2 = pd.DataFrame({'name':['foo', 'bar', 'baz']})
        con = Cartesian(reference=df2, relevance_threshold=None)
        con.fit_transform(df1)
           index_left name_left  index_right name_right relevance_score
0           0       foo            0        foo   {'name': 1.0}
1           0       foo            1        bar   {'name': 0.0}
2           0       foo            2        baz   {'name': 0.0}
3           1      bath            0        foo   {'name': 0.0}
4           1      bath            1        bar   {'name': 0.0}
5           1      bath            2        baz   {'name': 0.0}
    """

    def __init__(self, reference=None, relevance_func=None, relevance_threshold=None, *args, **kwargs):
        """
        start the comparator with the reference datafrane
        Args:
            reference (pd.DataFrame): reference data
            relevance_func: function used to compare the attributes col
            relevance_threshold (float): threshold on which to prune. None if no pruning.
            *args: N/A
            **kwargs: N/A
        """
        BaseConnector.__init__(self, *args, **kwargs)
        assert isinstance(reference, pd.DataFrame)
        if relevance_func is None:
            relevance_func = exact_score
        assert (callable(relevance_func))
        self.reference = reference
        self.attributesCols = self.reference.columns.tolist()
        self.relevanceCol = 'relevance_score'
        self.relevance_func = relevance_func
        self.relevance_threshold = relevance_threshold

    def transform(self, X, *args, **kwargs):
        """
        Calculate the combination between the input data and the reference data

        Args:
            X (pd.DataFrame): data frame containing the queries
            *args: N/A
            **kwargs: N/A

        Returns:
            pd.DataFrame

        Examples:
        from wookie.connectors import Cartesian
        df1 = pd.DataFrame({'name': ['foo', 'bath']})
        df2 = pd.DataFrame({'name':['foo', 'bar', 'baz']})
        con = Cartesian(reference=df2, relevance_threshold=None)
        con.fit_transform(df1)
           index_left name_left  index_right name_right relevance_score
0           0       foo            0        foo         {'name': 1.0}
1           0       foo            1        bar         {'name': 0.0}
2           0       foo            2        baz         {'name': 0.0}
3           1      bath            0        foo         {'name': 0.0}
4           1      bath            1        bar         {'name': 0.0}
5           1      bath            2        baz         {'name': 0.0}

        """
        product = cartesian_join(X, self.reference)
        product[self.relevanceCol] = product.apply(lambda row: self.relevance_score(row), axis=1)
        if self.relevance_threshold is None:
            return product
        else:
            # I put a [0] because I have only one relevancecols
            # TODO: mark it for several cols
            product = product[product[self.relevanceCol].apply(lambda r: self.pruning(r))]
            return product

    def fit(self, X, *_):
        """
        update the self.attributesCol attribute with the list of columns names that are common to both reference and input data
        Args:
            X (pd.DataFrame):
            *_:

        Returns:
            None
        """
        self.attributesCols = list(
            set(
                X.columns.tolist()
            ).intersection(
                set(
                    self.reference.columns.tolist()
                )
            )
        )
        return self

    def relevance_score(self, row):
        """
        calculate the relevance score for this particular row
        Args:
            row (pd.Series):

        Returns:
            dict
        """
        # TODO: show data
        score = dict(
            zip(
                self.attributesCols,
                map(
                    lambda c: self.relevance_func(row[c + '_left'], row[c + '_right']),
                    self.attributesCols
                )
            )
        )
        return score

    def pruning(self, relevance):
        """
        Returns boolean on whether or not to consider for future matches
        Args:
            relevance (dict):

        Returns:

        """
        assert isinstance(relevance, dict)
        a = sum(filter(None, relevance.values()))
        b = len(relevance)
        c = a / b > self.relevance_threshold
        return c
