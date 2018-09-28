import unicodedata

import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import partial_token_set_ratio as fuzzyscore

navalues = [
    '#', None, np.nan, 'None', '-', 'nan', 'n.a.',
    ' ', '', '#REF!', '#N/A', '#NAME?', '#DIV/0!',
    '#NUM!', 'NaT', 'NULL'
]
sepvalues = [' ', ',', '/', '-', ':', "'", '(', ')', '|', '°', '!', '\n', '_', '.']
companystopwords = [
    'aerospace',
    'ag',
    'and',
    'co',
    'company',
    'consulting',
    'corporation',
    'de',
    'deutschland',
    'dr',
    'electronics',
    'engineering',
    'europe',
    'formation',
    'france',
    'gmbh',
    'group',
    'hotel',
    'inc',
    'ingenierie',
    'international',
    'kg',
    'la',
    'limited',
    'llc',
    'ltd',
    'ltda',
    'management',
    'of',
    'oy',
    'partners',
    'restaurant',
    'sa',
    'sarl',
    'sas',
    'service',
    'services',
    'sl',
    'software',
    'solutions',
    'srl',
    'systems',
    'technologies',
    'technology',
    'the',
    'uk',
    'und'
]

streetstopwords = [
    'av'
    'avenue',
    'boulevard',
    'bvd'
    'calle',
    'place',
    'platz'
    'road',
    'rue',
    'str',
    'strasse',
    'strae',
    'via'
]
# Not used yet
endingwords = [
    'strasse',
    'str',
    'strae'
]

citystopwords = [
    'cedex',
    'po box',
    'bp'
]

# Not used yet
_bigcities = [
    'munich',
    'paris',
    'madrid',
    'hamburg',
    'toulouse',
    'berlin',
    'bremen',
    'london',
    'ulm',
    'stuttgart',
    'blagnac'
]
# Not used yet
_airbus_names = [
    'airbus',
    'casa',
    'eads',
    'cassidian',
    'astrium',
    'eurocopter'
]


def cleanduns(s):
    """
    Format the duns number
    Convert it to 9 digits str
    remove bad duns like DE0000000 or NDM999999, 999999999

    Args:
        s

    Returns:
        str: of length 9

    Examples:
        cleanduns(123456) --> '000123456'
        cleanduns('123456') --> '000123456'
        cleanduns(None) --> None
        cleanduns('NDM999999') --> None
    """
    s = idtostr(s, zfill=9, rmvwords=['999999999'])
    # remove bad duns like DE0000000 or NDM999999
    if s is None or s[2:] == '0000000' or s[:3] == 'NDM':
        return None
    else:
        return s


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


# ACRONYM TODO
# RMV STOPWORDS ENDING WORDS
# mergesapcols

# hasairbusname = lambda r: None if pd.isnull(r) else int(any(w in r for w in airbus_names))
# isbigcity = lambda r: None if pd.isnull(r) else int(any(w in r for w in bigcities))
# name_len = lambda r: None if pd.isnull(r) else len(r)
# id_cols = ['registerid', 'registerid1', 'registerid2', 'taxid', 'kapisid']

# cleandict = {
# 	'duns': cleanduns,
# 	'name': lowerascii,
# 	'street': lowerascii,
# 	'city': lowerascii,
# 	'name_wostopwords': (lambda r: nm.rmv_stopwords(r, stopwords=companystopwords), 'name'),
# 	'street_wostopwords': (lambda r: nm.rmv_stopwords(r, stopwords=streetstopwords, endingwords=endingwords), 'street'),
# 	'name_acronym': (lambda r: nm.acronym(r), 'name'),
# 	'postalcode': nm.format_int_to_str,
# 	'postalcode_1stdigit': (lambda r: None if pd.isnull(r) else str(r)[:1], 'postalcode'),
# 	'postalcode_2digits': (lambda r: None if pd.isnull(r) else str(r)[:2], 'postalcode'),
# 	'name_len': (lambda r: len(r), 'name'),
# 	'hasairbusname': (lambda r: 0 if pd.isnull(r) else int(any(w in r for w in airbus_names)), 'name'),
# 	'isbigcity': (lambda r: 0 if pd.isnull(r) else int(any(w in r for w in bigcities)), 'city')
#
# }
sapdict = {'SYSID': 'systemid',
           'LIFNR': 'companyid',
           'NAME1': 'name1',
           'NAME2': 'name2',
           'NAME3': 'name3',
           'NAME4': 'name4',
           'STRAS': 'streetaddress',
           'PSTLZ': 'postalcode',
           'ORT01': 'cityname',
           'LAND1': 'country',
           'KRAUS': 'dunsnumber',
           'STCD1': 'registerid1',
           'STCD2': 'registerid2',
           'STCEG': 'taxid',
           'VBUND': 'kapisid',
           'NATO_CAGE_CODE': 'cageid',
           'KTOKK': 'accounttype'}


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


def preparedf(data, ixname='ix'):
    """
    Perform several of pre-processing step on the dataframe
    - check the index
    - name: lower ascii, remove stopwords, create acronym
    - street: lower ascii, remove stopwords
    - city: lowerascii, remove stopwords
    - postalcode: lowerascii, first 2 chars
    - state: lowerascii
    - countrycode: lowerascii
    - duns: clean to str
    - ids: clean to str
    Args:
        data (pd.DataFrame): dataframe
    Returns:
        pd.DataFrame
    """
    df = data.copy()
    # check the index
    df = _chkixdf(df, ixname=ixname)

    # Index
    df[ixname] = df[ixname].apply(idtostr)

    ## Create an alert if the index is not unique
    if pd.Series(df[ixname]).unique().shape[0] != df.shape[0]:
        raise KeyError('Error: index is not unique')
    df.set_index([ixname], inplace=True)

    # Name
    df['name_ascii'] = df['name'].apply(lowerascii)
    df['name_ascii_wostopwords'] = df['name'].apply(
        lambda r: rmvstopwords(r, stop_words=companystopwords)
    )
    ## TODO acronym
    # df['name_acronym'] = df['name_ascii_wostopwords'].apply(acronym)

    # Location
    ## Street
    df['street_ascii'] = df['street'].apply(lowerascii)
    df['street_ascii_wostopwords'] = df['street_ascii'].apply(
        lambda r: rmvstopwords(r, stop_words=streetstopwords, ending_words=endingwords)
    )
    ## Postalcode
    df['postalcode_ascii'] = df['postalcode'].apply(
        lowerascii
    ).apply(
        idtostr
    )
    df['postalcode_2char'] = df['postalcode_ascii'].apply(
        lambda r: None if pd.isnull(r) else r[:2]
    )
    ## City
    df['city_ascii'] = df['city'].apply(
        lowerascii
    ).apply(
        lambda r: rmvstopwords(r, stop_words=citystopwords)
    )
    ## State
    if not 'state' in df.columns:
        df['state'] = None
    df['state_ascii'] = df['state'].apply(lowerascii)

    ## Country_code
    df['countrycode'] = df['countrycode'].apply(lowerascii)

    # IDs
    for c in ['duns', 'kapis', 'euvat', 'registerid', 'taxid', 'cage', 'siret', 'siren']:
        if not c in df.columns:
            df[c] = None
        df[c] = df[c].apply(idtostr)
    ## Duns
    df['duns'] = df['duns'].apply(cleanduns)

    return df


def _chkixdf(df, ixname='ix'):
    """
    Check that the dataframe does not already have a column of the name ixname
    And checks that the index name is ixname
    And reset the index to add ixname as a column
    Does not work on copy
    Args:
        df (pd.DataFrame):
        ixname (str): name of the index

    Returns:
        pd.DataFrame
    """
    if ixname in df.columns:
        raise KeyError('{} already in df columns'.format(ixname))
    else:
        df.reset_index(inplace=True, drop=False)
        if ixname not in df.columns:
            raise KeyError('{} not in df columns')
        return df
