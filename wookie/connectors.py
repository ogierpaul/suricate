import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from wookie import sbscomparators


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


class FuncTransfomer(TransformerMixin):
    """
    Transformer than can acco
    """

    def __init__(self, transfunc):
        """

        Args:
            transfunc (callable):
        """
        self.transfunc = transfunc

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        res = self.transfunc(X)
        return res


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
            relevance_func = sbscomparators._exact_score
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


def separatesides(df, lsuffix='_left', rsuffix='_right', y_true='y_true', ixname='ix'):
    """
    Separate a side by side training table into the left table, the right table, and the list of pairs
    Args:
        df (pd.DataFrame): side by side dataframe {['ix_left', 'ix_right'] :['name_left', 'name_right']}
        lsuffix (str): left suffix
        rsuffix (str): right suffix
        y_true (str): name of y_true column
        ixname (str): name in index column

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame
    """

    def takeside(df, suffix, ixname):
        new = df.copy().reset_index(drop=False)
        new = new[list(filter(lambda r: r[-len(suffix):] == suffix, new.columns))]
        new = rmvsuffix(new, suffix).drop_duplicates(subset=[ixname])
        new.set_index([ixname], inplace=True)
        return new

    xleft = takeside(df, lsuffix, ixname=ixname)
    xright = takeside(df, rsuffix, ixname=ixname)
    pairs = df[[y_true]].copy(
    )
    return xleft, xright, pairs


def createsbs(pairs, left, right, lsuffix='_left', rsuffix='_right', ixname='ix'):
    """
    Create a side by side table from a list of pairs (as a DataFrame)
    Args:
        pairs (pd.DataFrame): of the form {['ix_left', 'ix_right']:['y_true']}
        left (pd.DataFrame): of the form ['name'], index=ixname
        right (pd.DataFrame): of the form ['name'], index=ixname
        lsuffix (str): default '_left'
        rsuffix (str): default '_right'
        ixname (str): default 'ix' name of the index

    Returns:
        pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
    """
    xleft = _chkixdf(left.copy(), ixname=ixname)
    xright = _chkixdf(right.copy(), ixname=ixname)
    xpairs = pairs.copy().reset_index(drop=False)


    xleft = addsuffix(xleft, lsuffix).set_index(ixname + lsuffix)
    xright = addsuffix(xright, rsuffix).set_index(ixname + rsuffix)
    sbs = xpairs.join(
        xleft, on=ixname + lsuffix, how='left'
    ).join(
        xright, on=ixname + rsuffix, how='left'
    ).set_index(
        [
            ixname + lsuffix,
            ixname + rsuffix
        ]
    )
    return sbs


def safeconcat(dfs, usecols):
    """
    Concatenate two dataframe vertically using the same columns
    Checks that the indexes do not overlap
    safeconcat([df1,df2], usecols ['name']
    Args:
        dfs (list): [df1, df2] list of two dataframe
        usecols (list): list of column names

    Returns:
        pd.DataFrame {'ix': [cols]}
    """
    assert isinstance(dfs, list)
    df1 = dfs[0]
    assert isinstance(df1, pd.DataFrame)
    df2 = dfs[1]
    assert isinstance(df2, pd.DataFrame)

    # Check that the two indexes are of the same type
    for df in [df1, df2]:
        if df.index.dtype != np.dtype('O') or df.index.dtype in [np.dtype('float64'), np.dtype('int32')]:
            raise IndexError(
                'Non-string index type {}: all indexes should be string'.format(
                    type(df.index)
                )
            )

    intersection = np.intersect1d(df1.index, df2.index)
    if intersection.shape[0] > 0:
        raise KeyError('The indexes of the two df overlap: {}'.format(intersection))
    X = pd.concat(
        [
            df1[usecols],
            df2[usecols]
        ],
        axis=0,
        ignore_index=False
    )
    return X


def showpairs(pairs, left, right, usecols):
    """

    Args:
        pairs (pd.DataFrame): {[ix_left, ix_right]: [cols]}
        left (pd.DataFrame): {ix: [cols]}
        right (pd.DataFrame): {ix: [cols]}
        usecols (list): [name, duns, ..]

    Returns:
        pd.DataFrame: {[ix_left, ix_right]: [name_left, name_right, duns_left, duns_right]}
    """
    res = createsbs(pairs=pairs, left=left, right=right)
    displaycols = pairs.columns.tolist()
    for c in usecols:
        displaycols.append(c + '_left')
        displaycols.append(c + '_right')
    res = res[displaycols]
    return res


def concattrainnew(left, right, trainsbs, transfunc):
    """
    Args:
        left (pd.DataFrame): left data {ixname: [cols]}
        right (pd.DataFrame): right data {ixname: [cols]}
        trainsbs (pd.DataFrame): side_by_side analysis of the data \
            [ixname_lsuffix, ixname_rsuffix, col_lsuffix, col_rsuffix]
        transfunc (callable): preprocessing function

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: X_left {'ix': ['name']}, X_right, pairs {['ix_left', 'ix_right']: ['y_true']}
    """

    trainleft, trainright, pairs = separatesides(trainsbs)
    newleft = transfunc(left)
    newright = transfunc(right)
    trainleft = transfunc(trainleft)
    trainright = transfunc(trainright)
    usecols = list(set(trainleft.columns).intersection(set(newleft.columns)))
    X_left = safeconcat([trainleft, newleft], usecols=usecols)
    X_right = safeconcat([trainright, newright], usecols=usecols)
    return X_left, X_right, pairs


def _chkixdf(df, ixname='ix'):
    """
    Check that the dataframe does not already have a column of the name ixname
    And checks that the index name is ixname
    And reset the index to add ixname as a column
    Does not work on copy
    Args:
        df (pd.DataFrame): {ixname: [cols]}
        ixname (str): name of the index

    Returns:
        pd.DataFrame: [ixname, cols]
    """
    if ixname in df.columns:
        raise KeyError('{} already in df columns'.format(ixname))
    else:
        if df.index.name != ixname:
            raise KeyError('index name {} != expected name {}'.format(df.index.name, ixname))
        df.reset_index(inplace=True, drop=False)
        if ixname not in df.columns:
            raise KeyError('{} not in df columns'.format(ixname))
        return df


def rmv_end_str(w, s):
    """
    remove str at the end of tken
    :param w: str, token to be cleaned
    :param s: str, string to be removed
    :return: str
    """
    if w.endswith(s):
        w = w[:-len(s)]
    return w


def addsuffix(df, suffix):
    """
    Add a suffix to each of the dataframe column
    Args:
        df (pd.DataFrame):
        suffix (str):

    Returns:
        pd.DataFrame

    Examples:
        df.columns = ['name', 'age']
        addsuffix(df, '_left').columns = ['name_left', 'age_left']
    """
    df = df.copy().rename(
        columns=dict(
            zip(
                df.columns,
                map(
                    lambda r: r + suffix,
                    df.columns
                ),

            )
        )
    )
    assert isinstance(df, pd.DataFrame)
    return df


def rmvsuffix(df, suffix):
    """
    Rmv a suffix to each of the dataframe column
    Args:
        df (pd.DataFrame):
        suffix (str):

    Returns:
        pd.DataFrame

    Examples:
        df.columns = ['name_left', 'age_left']
        addsuffix(df, '_left').columns = ['name', 'age']
    """
    df = df.copy().rename(
        columns=dict(
            zip(
                df.columns,
                map(
                    lambda r: r[:-len(suffix)],
                    df.columns
                ),

            )
        )
    )
    assert isinstance(df, pd.DataFrame)
    return df
