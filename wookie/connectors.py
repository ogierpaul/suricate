import numpy as np
import pandas as pd

from wookie.preutils import _chkixdf, addsuffix, rmvsuffix


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


