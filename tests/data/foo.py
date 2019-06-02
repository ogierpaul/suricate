import pandas as pd
import pytest

from wookie.preutils import concatixnames


@pytest.fixture
def ix_names():
    # Create the ix_names variation

    ixname = 'ix'
    lsuffix = 'left'
    rsuffix = 'right'
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )
    names = dict()
    names['ixname'] = ixname
    names['ixnameleft'] = ixnameleft
    names['ixnameright'] = ixnameright
    names['ixnamepairs'] = ixnamepairs
    names['lsuffix'] = lsuffix
    names['rsuffix'] = rsuffix
    names['samplecol'] = 'name'
    return names


@pytest.fixture
def df_left(ix_names):
    """
    initiate a dataframe with just one column (samplecol) and three rows (foo, bar, ninja)
    Returns:
        pd.DataFrame
    """
    samplecol = ix_names['samplecol']
    ixname = ix_names['ixname']
    left = pd.DataFrame(
        {
            ixname: [0, 1, 2],
            samplecol: ['foo', 'bar', 'ninja']
        }
    ).set_index(ixname)
    return left


@pytest.fixture
def df_right(ix_names):
    """
    initiate a dataframe with just one column (samplecol) and three rows (foo, bar, baz)
    Args:
        ix_names:

    Returns:

    """
    samplecol = ix_names['samplecol']
    ixname = ix_names['ixname']
    right = pd.DataFrame(
        {
            ixname: [0, 1, 2],
            samplecol: ['foo', 'bar', 'baz']
        }
    ).set_index(ixname)
    return right


@pytest.fixture
def df_X(df_left, df_right):
    """
    initiate an array with two values, two sample dataframes left and right
    [df_left, df_right]
    Args:
        df_left (pd.DataFrame):
        df_right (pd.DataFrame):

    Returns:
        list
    """
    X = [df_left, df_right]
    return X


@pytest.fixture()
def df_sbs(ix_names):
    """
    Initiate the array of df_left and df_right side-by-side
    Args:
        ix_names:

    Returns:
        pd.DataFrame
    """
    # The side by side should be with None
    ixnameleft = ix_names['ixnameleft']
    ixnameright = ix_names['ixnameright']
    lsuffix = ix_names['lsuffix']
    rsuffix = ix_names['rsuffix']
    samplecol = ix_names['samplecol']
    sidebyside = pd.DataFrame(
        [
            [0, "foo", 0, "foo", 1],  # case equal
            [0, "foo", 1, "bar", 0],  # case False
            [0, "foo", 2, "baz", 0],
            [1, "bar", 0, "foo", 0],
            [1, "bar", 1, "bar", 1],  # case True for two
            [1, "bar", 2, "baz", 1],  # case True for two fuzzy
            [2, "ninja", 0, "foo", 0],
            [2, "ninja", 1, "bar", 0],
            [2, "ninja", 2, "baz", 0]
            #       [2, "ninja", None]  # case None --> To be removed --> No y_true
        ],
        columns=[ixnameleft, '_'.join([samplecol, lsuffix]), ixnameright, '_'.join([samplecol, rsuffix]), 'y_true']
    )
    sidebyside.set_index(ix_names['ixnamepairs'], inplace=True, drop=True)
    return sidebyside