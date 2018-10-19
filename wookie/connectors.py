import numpy as np
import pandas as pd

from wookie.preutils import addsuffix, rmvsuffix


def cartesian_join(left, right, lsuffix='left', rsuffix='right'):
    """

    Args:
        left (pd.DataFrame): table 1
        right (pd.DataFrame): table 2
        lsuffix (str):
        rsuffix (str):

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
            df (pd.DataFrame): {'ix':['name']}
            suffix (str): 'left'

        Returns:
            pd.DataFrame

        Examples:
            df = pd.DataFrame({'a':['foo', 'bar']})
                 a
            0   foo
            1	bar

            rename_with_suffix(df, 'right')

                index_right	a_right
            0	0	        foo
            1	1	        bar
        """
        if suffix is None:
            return df
        assert isinstance(suffix, str)
        assert isinstance(df, pd.DataFrame)
        df_new = df.copy()
        if df.index.name is None:
            ixname = 'index'
        else:
            ixname = df.index.name
        df_new.index.name = ixname
        df_new.reset_index(drop=False, inplace=True)
        cols = df_new.columns
        mydict = dict(
            zip(
                cols,
                map(lambda c: c + '_' + suffix, cols)
            )
        )
        df_new.rename(columns=mydict, inplace=True)
        return df_new

    # hack to create a column name unknown to both df1 and df2
    tempcolname = 'f1b3'
    while tempcolname in left.columns or tempcolname in right.columns:
        tempcolname += 'f'

    # create a new df1 with renamed cols
    df1new = rename_with_suffix(left, lsuffix)
    df2new = rename_with_suffix(right, rsuffix)
    df1new[tempcolname] = 0
    df2new[tempcolname] = 0
    dfnew = pd.merge(df1new, df2new, on=tempcolname).drop([tempcolname], axis=1)
    del df1new, df2new, tempcolname

    return dfnew


def separatesides(df, lsuffix='left', rsuffix='right', y_true_col='y_true', ixname='ix'):
    """
    Separate a side by side training table into the left table, the right table, and the list of pairs
    Args:
        df (pd.DataFrame): side by side dataframe {['ix_left', 'ix_right'] :['name_left', 'name_right']}
        lsuffix (str): left suffix 'left'
        rsuffix (str): right suffix 'right'
        y_true_col (str): name of y_true column
        ixname (str): name in index column

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series : {ix:['name'}, {'ix':['name'} {['ix_left', 'ix_right']:y_true}
    """

    def takeside(df, suffix, ixname):
        new = df.copy().reset_index(drop=False)
        new = new[list(filter(lambda r: r[-len(suffix):] == suffix, new.columns))]
        new = rmvsuffix(new, suffix).drop_duplicates(subset=[ixname])
        new.set_index([ixname], inplace=True)
        return new

    xleft = takeside(df, lsuffix, ixname=ixname)
    xright = takeside(df, rsuffix, ixname=ixname)
    pairs = df[y_true_col].copy(
    )
    pairs.name = y_true_col
    return xleft, xright, pairs


def createsbs(pairs, left, right, use_cols=None, lsuffix='left', rsuffix='right', ixname='ix'):
    """
    Create a side by side table from a list of pairs (as a DataFrame)
    Args:
        pairs (pd.DataFrame/pd.Series): of the form {['ix_left', 'ix_right']:['y_true']}
        left (pd.DataFrame): of the form ['name'], index=ixname
        right (pd.DataFrame): of the form ['name'], index=ixname
        use_cols (list): columns to use
        lsuffix (str): default 'left'
        rsuffix (str): default 'right'
        ixname (str): default 'ix' name of the index

    Returns:
        pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
    """
    ixnameleft = ixname + '_' + lsuffix
    ixnameright = ixname + '_' + rsuffix
    ixnamepairs = [ixnameleft, ixnameright]
    if use_cols is None or len(use_cols) == 0:
        use_cols = left.columns.intersection(right.columns)
    xleft = left[use_cols].copy().reset_index(drop=False)
    xright = right[use_cols].copy().reset_index(drop=False)
    xpairs = pairs.copy().reset_index(drop=False)
    if isinstance(xpairs, pd.Series):
        xpairs = pd.DataFrame(xpairs)
    xleft = addsuffix(xleft, lsuffix).set_index(ixnameleft)
    xright = addsuffix(xright, rsuffix).set_index(ixnameright)
    sbs = xpairs.join(
        xleft, on=ixnameleft, how='left'
    ).join(
        xright, on=ixnameright, how='left'
    ).set_index(
        ixnamepairs
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


def showpairs(pairs, left, right, use_cols=None, filterzeroes=False):
    """
    Like createsbs, but reorder the columns to compare left and right columns
    Args:
        pairs (pd.DataFrame/pd.Series): {[ix_left, ix_right]: col}
        left (pd.DataFrame): {ix: [cols]}
        right (pd.DataFrame): {ix: [cols]}
        use_cols (list): [name, duns, ..]
        filterzeroes (bool): 

    Returns:
        pd.DataFrame: {[ix_left, ix_right]: [name_left, name_right, duns_left, duns_right]}
    """
    if isinstance(pairs, pd.Series):
        xpairs = pd.DataFrame(pairs).copy()
    else:
        xpairs = pairs.copy()
    if use_cols is None:
        use_cols = left.columns.intersection(right.columns)
    res = createsbs(pairs=xpairs, left=left, right=right, use_cols=use_cols)
    displaycols = xpairs.columns.tolist()
    for c in use_cols:
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


def indexwithytrue(y_true, y_pred):
    """

    Args:
        y_true (pd.Series):
        y_pred (pd.Series):

    Returns:
        pd.Series: y_pred but with the missing indexes of y_true filled with 0
    """
    y_pred2 = pd.Series(index=y_true.index, name=y_pred.name)
    y_pred2.loc[y_true.index.intersection(y_pred.index)] = y_pred
    y_pred2.loc[y_true.index.difference(y_pred.index)] = 0
    return y_pred2


def _analyzeerrors(y_true, y_pred, rmvsameindex=True, ixnameleft='ix_left', ixnameright='ix_right'):
    ixnamepairs = [ixnameleft, ixnameright]
    y_true = y_true.copy()
    y_true.name = 'y_true'
    y_pred = y_pred.copy()
    y_pred.name = 'y_pred'
    pairs = pd.concat(
        [y_true, y_pred],
        axis=1
    )
    pairs['y_pred'].fillna(0, inplace=True)
    pairs = pairs.loc[
        ~(
            (pairs['y_pred'] == 0) & (
                pairs['y_true'] == 0
            )
        )
    ]
    pairs['correct'] = 'Ok'
    pairs.loc[
        (pairs['y_pred'] == 0) & (pairs['y_true'] == 1),
        'correct'
    ] = 'recall_error'
    pairs.loc[
        (pairs['y_pred'] == 1) & (pairs['y_true'] == 0),
        'correct'
    ] = 'precision_error'
    if rmvsameindex:
        pairs.reset_index(inplace=True)
        pairs = pairs[pairs[ixnameleft] != pairs[ixnameright]]
        pairs.set_index(ixnamepairs, inplace=True)
    return pairs
