import unicodedata

import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import partial_token_set_ratio as fuzzyscore

_suffixexact = 'exact'
_suffixtoken = 'token'
_suffixfuzzy = 'fuzzy'

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
    if s.endswith(('.0')):
        s = s[:-2]
    ## Remove special chars
    if not rmvchars is None:
        for c in rmvchars:
            s = s.replace(c, '')
    if len(s) == 0:
        return None
    ## Remove bad words
    if not rmvwords is None:
        if s in rmvwords:
            return None
    ## when done fill with leading zeroes
    if not zfill is None:
        s = s.zfill(zfill)
    return s


def lowerascii(s, lower=True):
    """
    Normalize to the ascii format
    Args:
        s (str): string to be formated
        lower (bool): to lower, default True
    Returns:
        str
    """
    """
    :param mystring: str
    :return: str, normalized as unicode
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


def rmv_end_list(w, mylist):
    """
    removed string at the end of tok
    :param w: str, word to be cleaned
    :param mylist: list, ending words to be removed
    :return: str
    """
    if type(mylist) == list:
        mylist.sort(key=len)
        for s in mylist:
            w = rmv_end_str(w, s)
    return w


sepvalues = [' ', ',', '/', '-', ':', "'", '(', ')', '|', '°', '!', '\n', '_', '.']


def split(mystring, seplist=sepvalues):
    """
    Split a string according to the list of separators
    if the string is an navalue, it returns None
    if the separator list is not provided, it used the default defined in the module
    (.sepvalues)
    Args:
        mystring (str): string to be split
        seplist (list): by default sepvalues, list of separators
    Returns:
        list
    """
    if mystring in navalues:
        return None
    else:
        if seplist is None:
            seplist = sepvalues
        for sep in seplist:
            mystring = mystring.replace(sep, ' ')
        mystring = mystring.replace('  ', ' ')
        mylist = mystring.split(' ')
        mylist = list(filter(lambda x: x not in navalues, mylist))
        mylist = list(filter(lambda x: len(x) > 0, mylist))
        return mylist


def concatenate_names(m):
    """
    This small function concatenate the different company names found across the names columns of SAP (name1, name2..)
    It takes the name found in the first column. If the name in the second column adds information to the first,
    it concatenates. And it continues like this for the other columns
    Args:
        m (list): list of strings
    Returns:
        concatenated list of strings
    Examples:
    name1='KNIGHT FRANK (SA) PTY LTD'
    name2='KNIGHT FRANK'
    name3='ex-batman'
    name4='kapis code 3000'
    concatenate_names([name1,name2,name3,name4]):
        'KNIGHT FRANK (SA) PTY LTD ex-batman kapis code 3000
    """
    # Remove na values
    # Remove na values
    var1 = ' '.join(filter(lambda c: not pd.isnull(c), m)).split(' ')
    if len(var1) == 0:
        return None
    res = var1[0]
    for ix in range(1, len(var1)):
        # Compare fuzzy matching score with already concatenated string
        rnew = var1[ix]
        score = fuzzyscore(res, rnew) / 100
        if pd.isnull(score) or score < 0.9:
            # if score is less than threshold add it
            res = ' '.join([res, rnew])
    return res


def rmvstopwords(myword, stop_words=None, ending_words=None):
    """
    remove stopwords, ending words, replace words
    Args:
        myword (str): word to be cleaned
        stop_words (list): list of words to be removed
        ending_words (list): list of words to be removed at the end of tokens
    Returns:
        str
    """

    if pd.isnull(myword):
        return None
    if len(myword) == 0:
        return None
    myword = lowerascii(myword)
    mylist = split(myword)
    mylist = [m for m in mylist if not m in stop_words]

    if ending_words is not None:
        newlist = []
        for m in mylist:
            newlist.append(rmv_end_list(m, ending_words))
        mylist = list(set(newlist)).copy()

    myword = ' '.join(mylist)
    myword = myword.replace('  ', ' ')
    myword = myword.lstrip().rstrip()

    if len(myword) == 0:
        return None
    else:
        return myword


navalues = [
    '#', None, np.nan, 'None', '-', 'nan', 'n.a.',
    ' ', '', '#REF!', '#N/A', '#NAME?', '#DIV/0!',
    '#NUM!', 'NaT', 'NULL'
]


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
