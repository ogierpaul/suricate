import pandas as pd

from wookie.preutils import idtostr

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


def preparedf(data):
    """
    Perform several of pre-processing step on the dataframe

    Args:
        data (pd.DataFrame): dataframe {'ix': cols}
    Returns:
        pd.DataFrame
    """
    df = data.copy()

    ## TODO acronym

    df['postalcode_2char'] = df['postalcode'].apply(
        lambda r: None if pd.isnull(r) else r[:2]
    )

    # IDs
    # for c in ['duns', 'kapis', 'euvat', 'registerid', 'taxid', 'cage', 'siret', 'siren']:
    #     if not c in df.columns:
    #         df[c] = None
    ## Duns
    df['duns'] = df['duns'].apply(cleanduns)

    return df
