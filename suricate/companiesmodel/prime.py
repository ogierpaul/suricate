import pandas as pd
import elasticsearch
# Suricate Model Building
from suricate.companiesmodel.standardmodel import companies_fit_predict
from suricate.data.companies import getsource, gettarget, getytrue



def get_prime_new_index(ix_con, ix_source_prime, ix_source_new, ix_target_prime, ix_target_new,
                        ixnamesource='ix_source', ixnametarget='ix_target'):
    """
    Select the pairs from ix_con where both the source and target index belongs only to the prime  and new datasets respectively
    Args:
        ix_con (pd.MultiIndex):
        ix_source_prime (pd.Index): Source Index of the Prime Dataset
        ix_source_new (pd.Index): Target Index of the New Dataset
        ix_target_prime (pd.Index): Source Index of the Prime Dataset
        ix_target_new (pd.Index): Target Index of the New Dataset
        ixnamesource (str): name of level source in the multi Index
        ixnametarget (str): name of level target in the multi Index

    Returns:
        pd.MultiIndex, pd.MultiIndex: ix_prime, ix_new, possible values of ix_con for the prime and new datasets
    Examples:
        df = pd.DataFrame([['a', 1], ['b', 26], ['z', 26]], columns=['ix_source', 'ix_target'])
        ix_con = df.set_index(keys=['ix_source', 'ix_target'], drop=True).index
        ix_source_prime = pd.Index(['a' ,'b'])
        ix_target_prime = pd.Index([1])
        ix_source_new = pd.Index(['z'])
        ix_target_new = pd.Index([26])
        ix_prime, ix_new = get_prime_new_index(ix_con, ix_source_prime, ix_source_new, ix_target_prime, ix_target_new, ixnamesource='ix_source', ixnametarget='ix_target')
        >> ix_prime
            ['a', 1]
        >> ix_new
            ['z', 26]
    """
    ixnamepairs = [ixnamesource, ixnametarget]
    df = pd.DataFrame(index=ix_con).reset_index(drop=False)
    ix_prime = df.loc[
        (df[ixnamesource].isin(ix_source_prime)) & (df[ixnametarget].isin(ix_target_prime))
        ].set_index(
        ixnamepairs
    ).index
    ix_new = df.loc[
        (df[ixnamesource].isin(ix_source_new)) & (df[ixnametarget].isin(ix_target_new))
        ].set_index(
        ixnamepairs
    ).index
    return ix_prime, ix_new




def check_index_no_overlap(prime_source, new_source, prime_target, new_target):
    """
    Checkes that the indexes of the prime and new datasets do not have duplicate values
    Args:
        prime_source (pd.DataFrame):
        new_source (pd.DataFrame):
        prime_target(pd.DataFrame):
        new_target (pd.DataFrame):

    Returns:
        bool
    """
    if len(new_target.index.intersection(prime_target.index)) > 0:
        raise IndexError(
            'Index of new_target and prime_target overlap on values: {}'.format(
                new_target.index.intersection(prime_target.index))
        )
    if len(new_source.index.intersection(prime_source.index)) > 0:
        raise IndexError(
            'Index of new_source and prime_source overlap on values: {}'.format(
                new_source.index.intersection(prime_source.index))
        )
    return True


def get_prime_data(usecols, nrows):
    prime_source = getsource(nrows=nrows)
    prime_target = gettarget(nrows=nrows)
    y_true = getytrue(Xst=[prime_source, prime_target])
    prime_source[usecols[0]] = prime_source.index
    prime_target[usecols[0]] = prime_target.index
    return prime_source, prime_target, y_true


def merge_prime_new(new_source, new_target, usecols, n_rows_prime):
    # Get Priming Data
    prime_source, prime_target, y_true = get_prime_data(usecols=usecols, nrows=n_rows_prime)

    # Concat the two datasets, source and prime
    mix_source = pd.concat([new_source[usecols], prime_source[usecols]], axis=0, ignore_index=False)
    mix_target = pd.concat([new_target[usecols], prime_target[usecols]], axis=0, ignore_index=False)
    ix_source_prime = prime_source.index
    ix_source_new = new_source.index
    ix_target_prime = prime_target.index
    ix_target_new = new_target.index
    check_index_no_overlap(prime_source, new_source, prime_target, new_target)
    return mix_source, mix_target, ix_source_prime, ix_source_new, ix_target_prime, ix_target_new, y_true


def separe_prime_new(df, ix_new_source, ix_new_target):
    ixnamepairs = list(df.columns)[:2]
    df.set_index(ixnamepairs, inplace=True)
    ix= df.index
    dfix = pd.DataFrame(index=ix)
    ixnamepairs = ix.names
    dfix.reset_index(drop=False,inplace=True)
    dfix = dfix.loc[
        (dfix[ixnamepairs[0]].isin(ix_new_source)) & (
            dfix[ixnamepairs[1]].isin(ix_new_target)
        )
        ]
    ix_new = dfix.set_index(ixnamepairs).index
    return df.loc[ix_new]

def prepare_new(d, usecols, marker_col):
    d = d.reset_index(drop=True)
    d.index.name = 'ix'
    d[marker_col] = 'new'
    d[usecols[0]] = d.index
    d = d[usecols]
    return d


def main():
    n_rows = 100
    index_name = 'prime'
    doc_type = index_name
    marker_col = 'origin'
    usecols = ['ix', 'name', 'street', 'city', 'postalcode', 'countrycode']
    new_source = getsource(nrows=n_rows)
    new_target = gettarget(nrows=None)
    new_source = prepare_new(new_source, usecols, marker_col)
    new_target = prepare_new(new_target, usecols, marker_col)
    mix_source, mix_target, ix_source_prime, ix_source_new, ix_target_prime, ix_target_new, y_true = merge_prime_new(new_source=new_source, new_target=new_target, usecols=usecols, n_rows_prime=None)
    es_client = elasticsearch.Elasticsearch()
    res = companies_fit_predict(df_source=mix_source, df_target=mix_target, y_true=y_true, es_client=es_client, doc_type=doc_type, index_name=index_name)
    res = separe_prime_new(res, ix_source_new, ix_target_new)
    return res

if __name__ == '__main__':
    res = main()
    res.to_csv('../../project/data_dir/extract_dir/matches.csv')




