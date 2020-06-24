import pytest
# from project.arp import Arp, clean_inputs
import pandas as pd
import datetime

raw_path = '../../project/data_dir/extract_dir/arp_mockaroo.csv'
nrows = 100
staging_pg = '../../project/data_dir/staging'
filename_pg = 'df_pg.csv'
filename_es = 'df_es.csv'
usecols = pd.Series(
    index= ['id', 'name', 'street', 'city', 'postalcode', 'state', 'country', 'iban', 'duns', 'ssn', 'ts'],
    data= ['arp', 'name', 'street', 'city', 'postalcode', 'state', 'countrycode', 'iban', 'duns', 'ssn', 'extract_ts']
)
ts_col = 'extract_ts'
coltypes = {
    'extract_ts': 'ts',
    'countrycode': 'str',
    'duns': 'str',
    'arp': 'str'
}
colzeroes = {
    'arp':6
}

### Transform
#### Data Cleansing
# - Select the cols
# - Rename the cols
# - Format the data to text, integer, etc..
# - Fill the leading zeroes
# - Remove the null or n.a. values
# - Remove duplicate
# - Remove sep and carriage return
# - Apply filters
# - Prevent JS injections



@pytest.fixture()
def raw_data():
    df = pd.read_csv(raw_path, sep=',', nrows=nrows)
    df['ts'] = datetime.datetime.now().strftime('%Y-%m-%d')
    return df

class TestArpExtract:
    def test_pandas_load_file(self, raw_data):
        df = raw_data
        assert isinstance(raw_data, pd.DataFrame)
        ts = pd.to_datetime(raw_data['ts'])
        assert nrows == df.shape[0]


class TestArpTransform:
    @pytest.fixture
    def arploader(self):
        a = Arp(
            usecols=usecols,
            dir_pg='staging',
            dir_es='staging'
        )
        return a

    def test_select_rename_cols(self, arploader, raw_data):
        df2 = arploader._selectrename(raw_data)
        assert set(df2.cols) == set(usecols.values())

    def format_data(self):
        return False


class TestInFlow():
    def test_main(self, raw_data):
        df2 = clean_inputs(df=raw_data, pkey='arp', usecols=usecols, sep_csv = '|', coltypes=coltypes, colzeroes=colzeroes,
                           transform_func=None)
        assert isinstance(df2, pd.DataFrame)

        # df_pg = a.transform_pg(df=df2)
        # df_es = a.transform_es(df=df2)
        # df_pg.to_csv(a.path_pg(), index=True, sep='|')
        # df_es.to_csv(a.path_es(), index=True, sep='|')
        # a.load_pg(path=a.path_pg())
        # a.load_es(path=a.load_es())
