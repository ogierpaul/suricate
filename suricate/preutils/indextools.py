import pandas as pd


def concatixnames(ixname='ix', lsuffix='left', rsuffix='right'):
    """

    Args:
        ixname (str): 'ix'
        lsuffix (str): 'left'
        rsuffix (str): 'right'

    Returns:
        str, str, list(): 'ix_left', 'ix_right', ['ix_left', 'ix_right']
    """
    ixnameleft = '_'.join([ixname, lsuffix])
    ixnameright = '_'.join([ixname, rsuffix])
    ixnamepairs = [ixnameleft, ixnameright]
    return ixnameleft, ixnameright, ixnamepairs


def chkixdf(df, ixname='ix'):
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
        addsuffix(df, 'left').columns = ['name_left', 'age_left']
    """
    df = df.copy().rename(
        columns=dict(
            zip(
                df.columns,
                map(
                    lambda r: r + '_' + suffix,
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
        suffix (str): 'left' (not _left) for coherency with the rest of the module

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
                    lambda r: r[:-(len(suffix) + 1)],
                    df.columns
                ),

            )
        )
    )
    assert isinstance(df, pd.DataFrame)
    return df


def createmultiindex(X, names=('ix_left', 'ix_right')):
    """

    Args:
        X(list): [df_left, df_right]
        names: ('ix_left', 'ix_right'))

    Returns:
        pd.MultiIndex
    """
    return pd.MultiIndex.from_product(
        [X[0].index, X[1].index],
        names=names
    )


def separatesides(df, ixname='ix', lsuffix='left', rsuffix='right', y_true_col='y_true'):
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

    # noinspection PyShadowingNames,PyShadowingNames
    def takeside(df, suffix, ixname):
        """

        Args:
            df (pd.DataFrame):
            suffix (str):
            ixname (str):

        Returns:

        """
        new = df.copy().reset_index(drop=False)
        new = new[list(filter(lambda r: r[-len(suffix):] == suffix, new.columns))]
        new = rmvsuffix(new, suffix).drop_duplicates(subset=[ixname])
        new.set_index([ixname], inplace=True)
        return new

    xleft = takeside(df=df, suffix=lsuffix, ixname=ixname)
    xright = takeside(df=df, suffix=rsuffix, ixname=ixname)
    pairs = df.loc[:, y_true_col].copy()
    return xleft, xright, pairs