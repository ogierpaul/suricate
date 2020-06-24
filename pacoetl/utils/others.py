import numpy as np
import pandas as pd
import datetime

def create_batch(df, batch_size=20000):
    """
    Separate the dataframe in a list of dataframes of fixed row size
    Args:
        df (pd.DataFrame): DataFrame
        batch_size (int): maximal size of dataframes

    Returns:
        list: list of pd.DataFrame
    """
    list_df = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
    return list_df


def uniquepairs(a, b):
    """
    create a unique combination of a and b by sorting and joining
    uniquepair(a,b) = uniquepairs(b,a)
    Args:
        a (str):
        b (str):

    Returns:
        str
    Examples:
        uniquepairs('foo',  'bar') --> 'bar-foo'
        uniquepairs ('bar', 'foo') --> 'bar-foo'
    """
    return '-'.join(sorted([str(a), str(b)]))


def deduperels(y, aggfunc=None):
    """
    From a vector with index as pair of records, return the unique pairs (not similar, not (a,b) and (b,a))
    Args:
        y (pd.Series):
        aggfunc (func):

    Returns:
        pd.Series: deduped series
    Examples:
        values_withdupes = [
        [0, 1, 0.9],
        [1, 0, 0.8],
        [0, 0, 0.5],
        [0, 2, 0.9],
        [2, 0, None],
        [1, 2, 0.9]
    ]
        values_expectedmean= [
            [0, 1, 0.85],
            [0, 2, 0.9],
            [1, 2, 0.9]
        ]
    """

    if aggfunc is None:
        f = np.nanmean
    else:
        f = aggfunc
    ixnamepairs = y.index.names
    colname = y.name
    df = pd.DataFrame(y).reset_index(drop=False)
    # Do not take pairs (a,a)
    df = df.loc[df[ixnamepairs[0]] != df[ixnamepairs[1]]]
    # Transform (a,b)-->'a-b' and (b, a) --> 'a-b'
    df['uniquepairs'] = df.apply(lambda r: uniquepairs(r[ixnamepairs[0]], r[ixnamepairs[1]]), axis=1)
    # Pivot to take the aggregate of the values from (b,a) and (a,b)
    y = df.pivot_table(index='uniquepairs', values=colname, aggfunc=f)
    y.columns = [colname]
    # Drop the duplicates
    z = df.drop_duplicates(subset=['uniquepairs']).drop([colname], axis=1)
    # Merge with the aggregate of values
    z = z.join(y, on='uniquepairs', how='left').drop(['uniquepairs'], axis=1).set_index(ixnamepairs)
    # Return the colname as series
    z = z[colname]
    return z


def printmessage(message):
    """
    print out a message prefixed by datetime.now()
    Args:
        message(str):

    Returns:
        None
    """
    print(datetime.datetime.now(), '| ', message)


def write_csv(df, staging_dir, fileprefix, sep='|'):
    """
    Write the dataframe as a csv file to staging dir
    Args:
        df (pd.DataFrame): with index
        staging_dir (str): directory where to write the file
        fileprefix (str): name of the file (will be suffixed by datetime.now() and .csv, see code).
        sep (str): delimiter

    Returns:
        str: filepath of the csv written
    """
    filename_timed = fileprefix + '_' + datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S") + '.csv'
    filepath = staging_dir + '/' + filename_timed
    df.to_csv(path_or_buf=filepath, encoding='utf-8', sep=sep, index=True)
    return filepath