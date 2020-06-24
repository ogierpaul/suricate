import pandas as pd
import datetime
from pacoetl.utils import clean_inputs
from pacoetl.utils.clean import rmv_arp_leading_zeroes, concat_cols, format_duns, format_arp, rmv_blank_values,  format_tax

pkey = 'arp'
usecols = pd.Series({'Airbus Vendor': 'arp',
                     'Airbus Vendor - Medium description': 'name',
                     'Name 2': 'name2',
                     'Name 3': 'name3',
                     'House number and street': 'street',
                     'Street 2': 'street2',
                     'Street  4': 'street4',
                     'Postal Code': 'postalcode',
                     # 'CAM: PO Box Postal Code': 'pobox_postalcode',
                     'Location': 'city',
                     # 'CAM: PO Box Location': 'pobox_city',
                     'Region (State, Province, County)': 'state',
                     'Country': 'countrycode',
                     'VAT Code': 'eu_vat',
                     'Tax number 1': 'tax1',
                     'Tax number 2': 'tax2',
                     'Tax Number at Responsible Tax Authority': 'tax3',
                     'D-U-N-S Number': 'duns',
                     'CAGE Code': 'cage',
                     'Harmonize Supplier Name': 'arp_harmonizedname',
                     'Partner company number': 'arp_partnercompany'})
coltypes = {}
colzeroes = {'arp': 6, 'duns': 9}
ordercols = [
        'arp',
        'name',
        'street',
        'postalcode',
        'city',
        'state',
        'countrycode',
        'duns',
        'eu_vat',
        'tax1',
        'tax2',
        'tax3',
        'arp_harmonizedname',
        'arp_partnercompany',
        'cage',
        'concatenatedids'
    ]


def clean(df):
    def select_valid_records(df):
        """
        Select only valid ARP 'WHERE...'
        In this version I do not do any filtering
        Args:
         df (pd.DataFrame): dataframe

        Returns:
         pd.DataFrame
        """
        return df

    def transform_arp(df):
        """
        Chain the transforms
        Args:
            df (pd.DataFrame): with index

        Returns:
            pd.DataFrame
        """
        idcols =['eu_vat', 'tax1', 'tax2', 'tax3']

        print(datetime.datetime.now(), ' | Inputs sanitized')
        df = select_valid_records(df=df)
        print(datetime.datetime.now(), ' | Select valid records')
        df = rmv_arp_leading_zeroes(df=df, usecol='arp')
        print(datetime.datetime.now(), ' | Removed arp leading zeroes')
        df = rmv_blank_values(df=df)
        print(datetime.datetime.now(), ' | Removed blank values')
        df = concat_cols(df=df, colname='name', cols=['name', 'name2', 'name3'])
        print(datetime.datetime.now(), ' | Concat col name')
        df = concat_cols(df=df, colname='street', cols=['street', 'street2', 'street4'])
        print(datetime.datetime.now(), ' | Concat col street')
        for c in idcols:
            df = format_tax(df, c, 'countrycode')
        df = concat_cols(df=df, colname='concatenatedids', cols=['eu_vat', 'tax1', 'tax2', 'tax3'], sep=';')
        print(datetime.datetime.now(), ' | Concat col ids')
        df = format_duns(df=df, col='duns')
        print(datetime.datetime.now(), ' | Format duns')
        df = format_arp(df=df, col='arp')
        print(datetime.datetime.now(), ' | Format arp')
        df.dropna(subset=['arp'], inplace=True)
        print(datetime.datetime.now(), ' | Dropna arp')
        df.drop(['street2', 'street4', 'name2', 'name3', ], axis=1, inplace=True)
        df = df[ordercols]
        df.set_index('arp', inplace=True)
        return df

    df2 = clean_inputs(df=df, pkey=pkey, usecols=usecols, sep_csv='|', coltypes=coltypes, colzeroes=colzeroes,
                       transform_func=transform_arp)

    return df2


