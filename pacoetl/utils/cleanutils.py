import pandas as pd
import numpy as np
import unicodedata
import bleach


def idtostr(var1, zfill=None, rmvlzeroes=True, rmvchars=None, rmvwords=None):
    """
    Format an id to a str, removing separators like [, . / - ], leading zeros

    Args:
        var1: id to be formatted, scalar
        zfill (int): length of expected string, optional, default None
        rmvlzeroes (bool): remove leading zeroes, default True
        rmvchars (list): list of chars to be removed, default ['-', '.', ' ', '/', '#']
        rmvwords (list): list of values to be removed after cleansing, default ['#', 'None'])

    Returns:
        str

    Examples:
        - convert an int or float to str
            idtostr(123) --> '123'
        - convert str to str
            idtostr('foo') --> 'foo'
        - returns None when input is null
            idtostr(None) --> None
        - left strip leading zeroes
            idtostr('0123') --> 123
        - but can also zero padd when called
            idtostr('0123', zfill=6) --> '000123'
        - remove chars
            idtostr('1-23', rmvchars=['-']) --> '123'
        - remove .0:
            idtostr(123.0) --> '123'
        - remove words
            idtostr('#N/A', rmvwords=['#N/A']) --> None
    """
    if pd.isnull(var1):
        return None
    if not type(var1) in (str, float, int):
        raise TypeError(
            '{} of type {} not in str, float, int, None'.format(var1, type(var1))
        )
    # set default values for rmvchars and rmvwords
    if rmvchars is None:
        rmvchars = ['-', '.', ' ', '/', '#']
    if rmvwords is None:
        rmvwords = ['#', 'None']

    # convert to string
    s = str(var1)

    # Trim string
    ## remove leading zeroes
    if rmvlzeroes is True:
        s = s.lstrip('0')

    # remove trailing zero decimals
    if s.endswith('.0'):
        s = s[:-2]

    ## Remove special chars
    if rmvchars is not None:
        for c in rmvchars:
            s = s.replace(c, '')
    if len(s) == 0:
        return None
    ## Remove bad words
    if rmvwords is not None:
        if s in rmvwords:
            return None
    ## when done fill with leading zeroes
    if zfill is not None:
        s = s.zfill(zfill)
    return s


navalues = [
    '#', None, np.nan, 'None', '-', 'nan', 'n.a.',
    ' ', '', '#REF!', '#N/A', '#NAME?', '#DIV/0!',
    '#NUM!', 'NaT', 'NULL'
]


def lowerascii(s, lower=True):
    """
    Normalize to the ascii format
    Args:
        s (str): string to be formated
        lower (bool): to lower, default True
    Returns:
        str
    """
    if s in navalues:
        return None
    else:
        s = str(s)
        s = unicodedata.normalize('NFKD', s)
        s = s.encode('ASCII', 'ignore').decode('ASCII')
        if lower is True:
            s = s.lower()
        return s


def castcols(df, cols):
    """
    For each (key, value) pair (k, v); cast the columns k as the data type v
    v should be one of those int, float, str, bool, ts
    Args:
        df (pd.DataFrame):
        cols (dict):
    Returns:
        pd.DataFrame
    """
    for (k, v) in cols.items():
        if v == 'int':
            df[k] = df[k].astype(int)
        elif v == 'float':
            df[k] = df[k].astype(float)
        elif v == 'str':
            df[k] = df[k].astype(str)
        elif v == 'bool':
            df[k] = df[k].astype(bool)
        elif v == 'ts':
            df[k] = pd.to_datetime(df[k])
        else:
            raise ValueError(v, "Not in list [int, float, str, bool, ts]")
    return df


def zeropadding(df, cols):
    """
    for
    Args:
        df (pd.DataFrame):
        cols (dict)

    Returns:
        pd.DataFrame
    """
    for (k, v) in cols.items():
        df[k] = df[k].str.zfill(v)
    return df


def convertnone(s, navalues=navalues):
    if not s in navalues:
        return s
    else:
        return None


def sanitize_js(i):
    """
    Sanitize the input to prevent JS injection attack with bleach.
    Args:
        i: value to be cleaned
    """
    if isinstance(i, str):
        return bleach.clean(i)
    else:
        return i


def sanitize_csv(i, sep=','):
    """
    Sanitize the inputs to prevent csv encoding errors: will remove carriage return, new lines, separator
    Args:
        i:
        sep (str): separator to avoid for csv output

    Returns:

    """
    if isinstance(i, str):
        for s in ['\n', '\r', sep, '"', "'", '  ', '  ']:
            i = i.replace(s, ' ')
        return i.strip()
    else:
        return i


def primary_key_check(df, key):
    """
    Remove from the dataframe all rows where the primary key is null
    And where the key is duplicate
    Args:
        df (pd.DataFrame): input data
        key (str/list): name of the primary key column (or list)

    Returns:
        pd.DataFrame
    """
    if type(key) == str:
        key_cols = [key]
    else:
        key_cols = key
    return df.dropna(axis=0, subset=key_cols).drop_duplicates(subset=key_cols)


def select_rename_order_cols(df, cols):
    """
    - Select the columns to be inserted
    - Rename them
    - Re-order the attributes as in the destination table

    Args:
        df (pd.DataFrame): data to be loaded
        cols (pd.Series): ordered dict containing as value the list of output cols needed, as index their name \
        in the source data

    Returns:
        pd.DataFrame
    """
    assert isinstance(cols, pd.Series)
    old_cols = cols.index
    new_cols = cols.values
    df2 = df[old_cols]  # select interesting cols from raw data
    df2 = df2.rename(columns=cols)  # rename them
    df2 = df2[new_cols]  # re-order cols
    return df2


def _format_pkey(pkey):
    """
    Format the primary key to be inserted into an SQL query
    Args:
        pkey (str/list): primary key

    Returns:
        str: pkey: column name or list of column names separated by comma
    """
    if isinstance(pkey, str):
        pkey_s = pkey
    else:
        pkey_s = ', '.join(pkey)
    return pkey_s

def drop_na_dupes(df, subset):
    """
    Will drop null values or duplicates values for the subset provided
    Args:
        df (pd.DataFrame):
        subset (str/list): column label or list of column lables

    Returns:
        pd.DataFrame
    """
    if isinstance(subset, str):
        cols = [subset]
    else:
        cols = subset
    return df.dropna(subset=cols).drop_duplicates(subset=cols)

def validate_cols(cols, usecols=None, colzeroes=None, coltypes=None, pkey=None):
    """
    Check that the column names provided in usecols are in line with source cols.
    And that colzeroes, coltypes, pley are in line with the column names renamed from usecols
    Args:
        cols (set): source dataframe columns
        usecols (pd.Series):
        colzeroes (dict):
        coltypes (dict):
        pkey (str/list):

    Returns:
        bool
    Raises
        ValueError
    """
    if usecols is None:
        old = set(cols)
        new = set(cols)
    else:
        old = set(usecols.index)
        new = set(usecols.values)
    missing_labels = old.difference(cols)
    if len(missing_labels) > 0:
        ValueError(missing_labels, "Cols not found in source DataFrame")

    for d in colzeroes, coltypes:
        if d is not None:
            keys = set(d.keys())
            missing_labels = keys.difference(new)
            if len(missing_labels) > 0:
                ValueError(missing_labels, "Cols from colzeroes or coltypes not found in renamed DataFrame")
    if pkey is not None:
        if isinstance(pkey, str):
            pkey_s = set([pkey])
        else:
            pkey_s = set(pkey)
        missing_labels = pkey_s.difference(new)
        if len(missing_labels) > 0:
            ValueError(missing_labels, "Cols from Pkey not found in renamed DataFrame")

    return True

def clean_inputs(df, clean_js=True, clean_csv=True, coltypes=None, colzeroes=None,
                 pkey=None, transform_func=None, removena=True, usecols=None, sep_csv=',',
                 ):
    """
    Clean the data frame by performing the following steps:
        - select and rename
        - cast the right columns
        - convert na values to None
        - zero padding
        - sanitize for JS inputs
        - remove confusing values for csv output
        - drop null duplicates on pkey
        - transform the dataframe for ad-hoc transformation (filtering, etc..)
    Args:
        df (pd.DataFrame):
        clean_js (bool): if True, call sanitize_js to prevent JavaScript injection Attacks
        clean_csv (bool): if True, call sanitize_csv to prevent confusing csv output
        coltypes (dict): if not None,call castcols to cast the columns to the right data type
        colzeroes (dict): if not None, call zeropadding to padd certain columns to a right number of digies
        pkey (str/list): if not None, call drop_na_duptes to drop null and duplicate values for the primary key
        transform_func (func): if not None, will be called to do further transformation on the DF. \
            takes as input a DF, returns a DF. Can be used for filtering.
        removena (bool): if True, call convertnone to convert possible na values to None
        sep_csv (str):  separator to be removed from column content in sanitize_csv.
        usecols (pd.Series): if not None, used as an ordered dict for select_rename_order columns

    Returns:
        pd.DataFrame
    """
    df2 = df.copy()
    validate_cols(cols=df.columns, usecols=usecols, colzeroes=colzeroes, coltypes=coltypes, pkey=pkey)
    if usecols is not None:
        df2 = select_rename_order_cols(df=df2, cols=usecols)
    if coltypes is not None:
        df2 = castcols(df=df2, cols=coltypes)
    if removena is True:
        df2 = df2.applymap(convertnone)
    if colzeroes is not None:
        df2 = zeropadding(df=df2, cols=colzeroes)
    if clean_js is True:
        df2 = df2.applymap(sanitize_js)
    if clean_csv is True:
        df2 = df2.applymap(sanitize_csv)
    if removena is True:
        df2 = df2.applymap(convertnone)
    if pkey is not None:
        df2 = drop_na_dupes(df=df2, subset=pkey)
    if transform_func is not None:
        df2 = transform_func(df2)
    return df2
