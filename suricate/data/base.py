import pandas as pd
from os.path import dirname, join

from suricate.preutils import concatixnames
import suricate.data.companydata

def create_path(filename, foldername):
    """
    Args:
        filename (str): name of csv file to read
        foldername (str): name of enclosing folder inside of suricate.data

    Returns:
        path to csv file

    Examples:
        create_path(filename='left.csv', foldername='csv_company')
           suricate_root/suricate/data/csv_company/left.csv
    """
    module_path = dirname(__file__)
    filepath = join(*[module_path, foldername, filename])
    return filepath

def open_csv(filename, foldername, encoding='utf-8', sep=',', index_col=None, nrows=None):
    """
    Read the file and return
    Args:
        filename (str): name of csv file
        foldername (str): name of enclosing suricate folder in suricate.data
        encoding (str):
        sep (str):
        index_col:
        nrows (int):

    Returns:
        pd.DataFrame
    """
    filepath = create_path(filename=filename, foldername=foldername)
    df = pd.read_csv(
        filepath_or_buffer=filepath,
        sep=sep,
        index_col = index_col,
        encoding=encoding,
        nrows=nrows
    )
    return df


def _init_ixnames():
    """
    {
        'ixname': 'ix,
        'ixnameleft': 'ix_left'
        'ixnameright': 'ix_right'
        'ixnamepairs': ['ix_left', 'ix_right']
        'lsuffix': 'left'
        'rsuffix': 'right'
    }
    Returns:
        dict
    """
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
    return names

ix_names = _init_ixnames()