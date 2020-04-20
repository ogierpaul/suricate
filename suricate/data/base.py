import pandas as pd
from os.path import dirname, join

from suricate.preutils import concatixnames

def create_path(filename, foldername):
    """
    Args:
        filename (str): name of csv file to read
        foldername (str): name of enclosing folder inside of suricate.data

    Returns:
        path to csv file

    Examples:
        create_path(filename='source.csv', foldername='csv_company')
           suricate_root/suricate/data/csv_company/source.csv
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
        'ixnamesource': 'ix_source'
        'ixnametarget': 'ix_target'
        'ixnamepairs': ['ix_source', 'ix_target']
        'source_suffix': 'left'
        'target_suffix': 'right'
    }
    Returns:
        dict
    """
    ixname = 'ix'
    source_suffix = 'source'
    target_suffix = 'target'
    ixnamesource, ixnametarget, ixnamepairs = concatixnames(
        ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix
    )
    names = dict()
    names['ixname'] = ixname
    names['ixnamesource'] = ixnamesource
    names['ixnametarget'] = ixnametarget
    names['ixnamepairs'] = ixnamepairs
    names['source_suffix'] = source_suffix
    names['target_suffix'] = target_suffix
    return names

ix_names = _init_ixnames()