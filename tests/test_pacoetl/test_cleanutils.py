from paco.utils import sanitize_js, sanitize_csv, convertnone, clean_inputs
from paco.utils.cleanutils import validate_cols
import pytest
import pandas as pd
import datetime

@pytest.fixture
def raw_data_types():
    y = pd.Series(
        data=['foo',
              'bar,',
              """
              baz
              
              
              
              """,
              None,
              '#N/A',
              '',
              "an <script>evil()</script> example",
              1,
              2,
              0.5,
              False,
              True
              ],
        name='test'
    )
    return y


def test_sanitize_csv(raw_data_types):
    assert sanitize_csv('foo,', sep=',') == 'foo'
    x = raw_data_types.apply(sanitize_csv)
    assert x.loc[1] == 'bar'
    assert x.loc[2].strip() == 'baz'

def test_convertnone(raw_data_types):
    x = raw_data_types.apply(convertnone)
    for i in [3, 4, 5]:
        assert x.loc[i] is None

def test_sanitize_js(raw_data_types):
    x = raw_data_types.apply(sanitize_js)
    assert isinstance(x.loc[6], str)
    assert isinstance(x.loc[7], int)
    assert x.loc[0] == 'foo'
    assert x.loc[6] != raw_data_types.loc[6]

def test_validate_cols():
    df = pd.DataFrame(columns=['constant', 'old', 3])
    cols = df.columns
    usecols = pd.Series(
        index=['constant', 'old'],
        data=['constant', 'new']
    )
    assert validate_cols(cols=cols, usecols=usecols)
    coltypes = {'constant': 'str'}
    colzeroes = {'new': 9}
    assert validate_cols(cols=cols, usecols=usecols, coltypes=coltypes, colzeroes=colzeroes)
    assert validate_cols(cols=cols, usecols=usecols, pkey='constant')
    assert validate_cols(cols=cols, pkey='constant')
    assert validate_cols(cols=cols, usecols=usecols, pkey=['constant', 'new'])
    try:
        usecols = pd.Series(
            index=['constant', 'wrong'],
            data=['constant', 'new']
        )
        validate_cols(cols=cols, usecols=usecols)
    except ValueError:
        assert True
    try:
        usecols = pd.Series(
            index=['constant', 'old'],
            data=['constant', 'new']
        )
        coltypes = {'wrong': 'str'}
        validate_cols(cols=cols, usecols=usecols, coltypes=coltypes)
    except ValueError:
        assert True
    try:
        usecols = pd.Series(
            index=['constant', 'old'],
            data=['constant', 'new']
        )
        pkey = 'wrong'
        validate_cols(cols=cols, usecols=usecols, pkey=pkey)
    except ValueError:
        assert True

@pytest.fixture()
def raw_data():
    raw_path = 'extract_dir/arp.csv'
    nrows = 100
    df = pd.read_csv(raw_path, sep=',', nrows=nrows)
    df['ts'] = datetime.datetime.now().strftime('%Y-%m-%d')
    return df

def test_clean_inputs(raw_data):
    def arp_filter(df):
        """
        Dummy filter and transform function for df
        Args:
            df (pd.DataFrame)
        Returns:
            pd.DataFrame
        """
        df2 = df.loc[df['countrycode'] != 'ZA']
        df2['duns'] = df2['duns'].str.replace('-', '')
        return df2
    usecols = pd.Series(
        index=['id', 'name', 'street', 'city', 'postalcode', 'state', 'country', 'iban', 'duns', 'ssn', 'ts'],
        data=['arp', 'name', 'street', 'city', 'postalcode', 'state', 'countrycode', 'iban', 'duns', 'ssn',
              'extract_ts']
    )
    coltypes = {
        'extract_ts': 'ts',
        'countrycode': 'str',
        'duns': 'str',
        'arp': 'str'
    }
    colzeroes = {
        'arp': 6
    }
    df2 = clean_inputs(df=raw_data, pkey='arp', usecols=usecols, sep_csv='|', coltypes=coltypes, colzeroes=colzeroes,
                       transform_func=arp_filter)
    assert isinstance(df2, pd.DataFrame)