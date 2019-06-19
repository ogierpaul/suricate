import pandas as pd
from os.path import dirname, join


def create_path(filename, foldername):
    """
    Args:
        filename (str): name of csv file to read
        foldername (str): name of enclosing folder inside of suricate.data

    Returns:
        path to csv file

    Examples:
        create_path(filename='left.csv', foldername='datacsv')
           suricate_root/suricate/data/datacsv/left.csv
    """
    module_path = dirname(__file__)
    filepath = join(*[module_path, foldername, filename])
    return filepath

def open_csv(filepath, encoding='utf-8', sep=',', index_col=None, nrows=None):
    df = pd.read_csv(
        filepath_or_buffer=filepath,
        sep=sep,
        index_col = index_col,
        encoding=encoding,
        nrows=nrows
    )
    return df
