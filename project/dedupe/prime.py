import pandas as pd
import elasticsearch
# Suricate Model Building
from suricate.companiesmodel.standardmodel import companies_fit_predict
from suricate.data.companies import getsource, gettarget, getytrue
import psycopg2
from sqlalchemy import create_engine


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
    """
    Obtain the source, target and y_true dataframe
    Args:
        usecols (list): columns to use ['ix', 'name', 'street', 'city', 'postalcode', 'countrycode', 'duns']
        nrows (int): number of rows to read

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series
    """
    prime_source = getsource(nrows=nrows)
    prime_target = gettarget(nrows=nrows)
    y_true = getytrue(Xst=[prime_source, prime_target])
    # prime_source[usecols[0]] = prime_source.index
    # prime_target[usecols[0]] = prime_target.index
    return prime_source, prime_target, y_true


def concat_prime_new(prime_source, prime_target, new_source, new_target, usecols):
    """
    Concatenate the prime and new datasets
    Args:
        prime_source (pd.DataFrame):
        prime_target (pd.DataFrame):
        new_source (pd.DataFrame):
        new_target (pd.DataFrame):
        usecols (list):

    Returns:
        pd.DataFrame, pd.DataFrame
    """
    # Concat the two datasets, source and prime
    mix_source = pd.concat([new_source[usecols], prime_source[usecols]], axis=0, ignore_index=False)
    mix_target = pd.concat([new_target[usecols], prime_target[usecols]], axis=0, ignore_index=False)
    check_index_no_overlap(prime_source, new_source, prime_target, new_target)
    return mix_source, mix_target


def separe_prime_new(df, ix_new_source, ix_new_target):
    """
    From the results datasets, separate the pairs that contain exclusively elements from new datasets
    Args:
        df (pd.DataFrame):
        ix_new_source (pd.Index):
        ix_new_target (pd.Index):

    Returns:
        pd.DataFrame: Results with only new pairs
    """
    #ixnamepairs = list(df.columns)[:2]

    # df.set_index(ixnamepairs, inplace=True)
    ixnamepairs = df.index.names
    ix = df.index
    dfix = pd.DataFrame(index=ix)
    dfix.reset_index(drop=False, inplace=True)
    dfix = dfix.loc[
        (dfix[ixnamepairs[0]].isin(ix_new_source)) & (
            dfix[ixnamepairs[1]].isin(ix_new_target)
        )
        ]
    ix_new = dfix.set_index(ixnamepairs).index
    return df.loc[ix_new]


def prepare_new(d, usecols):
    """

    Args:
        d (pd.DataFrame): dataframe with index
        usecols:

    Returns:
        pd.DataFrame: dataframe with a copy of the index as column, and the index name 'ix'
    """
    # d = d.reset_index(drop=True)
    d.index.name = 'ix'
    # d[usecols[0]] = d.index
    d = d[usecols]
    return d


def prime_possible_matches(new_source, new_target, nrows=None, n_estimators=500):
    """
    the list of possible pairs using priming dataset
    Args:
        new_source (pd.DataFrame):
        new_target (pd.DataFrame):
        nrows (int)
    Returns
        pd.DataFrame
    """
    index_name = 'prime'
    doc_type = index_name
    usecols = ['name', 'street', 'city', 'postalcode', 'countrycode']
    new_source = prepare_new(new_source, usecols)
    new_target = prepare_new(new_target, usecols)
    prime_source, prime_target, y_true = get_prime_data(usecols=usecols, nrows=nrows)
    mix_source, mix_target, = concat_prime_new(
        new_source=new_source, new_target=new_target, prime_source=prime_source, prime_target=prime_target,usecols=usecols)
    es_client = elasticsearch.Elasticsearch()
    print('data concatenated')
    res = companies_fit_predict(df_source=mix_source, df_target=mix_target, y_true=y_true, es_client=es_client,
                                doc_type=doc_type, index_name=index_name, n_estimators=n_estimators)
    res = separe_prime_new(res, new_source.index, new_target.index)
    return res


def _test():
    n_rows = None
    index_name = 'prime'
    doc_type = index_name
    usecols = ['ix', 'name', 'street', 'city', 'postalcode', 'countrycode']
    all_data = pd.concat([getsource(nrows=None), gettarget(nrows=None)], axis=0, ignore_index=False)
    new_source = all_data
    new_target = all_data
    new_source = prepare_new(new_source, usecols)
    new_target = prepare_new(new_target, usecols)
    prime_source, prime_target, y_true = get_prime_data(usecols=usecols, nrows=n_rows)
    mix_source, mix_target, = concat_prime_new(
        new_source=new_source, new_target=new_target, usecols=usecols)
    es_client = elasticsearch.Elasticsearch()
    res = companies_fit_predict(df_source=mix_source, df_target=mix_target, y_true=y_true, es_client=es_client,
                                doc_type=doc_type, index_name=index_name)
    res = separe_prime_new(res, new_source.index, new_target.index)
    engine = create_engine('postgresql://suricateeditor:66*$%HWqx*@localhost:5432/suricate')
    res.to_sql(name='results', con=engine, if_exists='replace', index=True)
    new_source[usecols].to_sql(name='df_source', con=engine, if_exists='replace', index=False)
    new_target[usecols].to_sql(name='df_target', con=engine, if_exists='replace', index=False)
    R2 = res.reset_index(drop=False)

    # R2 = res.reset_index(drop=False)
    # sample = R2.loc[R2['ix_source'].isin(id_cluster_points)]
    # res.to_csv('../../project/data_dir/extract_dir/matches.csv')
