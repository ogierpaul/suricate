import pandas as pd
import elasticsearch
# Suricate Model Building
from suricate.companiesmodel.standardmodel import prime_model
from suricate.data.companies import getsource, gettarget, getytrue
# Sci-kit-learn Model Building
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from suricate.sbstransformers import SbsApplyComparator


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

def main():
    n_rows = 650
    n_estimators = 500
    n_hits_max = 10
    index_name = 'prime'
    doc_type = index_name
    marker_col = 'origin'
    usecols = ['ix', 'name', 'street', 'city', 'postalcode', 'countrycode']
    new_source = getsource(nrows=None).tail(n=n_rows)
    new_source[marker_col] = 'new'
    new_target = gettarget(nrows=None).tail(n=n_rows)
    new_target[marker_col] = 'new'
    new_source[usecols[0]] = new_source.index
    new_target[usecols[0]] = new_target.index

    sbs_score_list = [
        ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
        ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
        ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
        ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple'))
    ]

    scoreplan = {
        'name': {
            'type': 'FreeText'
        },
        'street': {
            'type': 'FreeText'
        },
        'city': {
            'type': 'FreeText'
        },
        'duns': {
            'type': 'Exact'
        },
        'postalcode': {
            'type': 'FreeText'
        },
        'countrycode': {
            'type': 'Exact'
        }
    }



    mapping = {
        "mappings": {
            doc_type: {
                "properties": {
                    "ix": {"type": "keyword"},
                    "name": {"type": "text"},
                    "street": {"type": "text"},
                    "city": {"type": "text"},
                    "postalcode": {"type": "text"},
                    "countrycode": {"type": "keyword"}
                }
            }
        }
    }
    pipemodel = Pipeline(steps=[
        ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaler', Normalizer()),
        ('PCA', PCA(n_components=3)),
        ('Predictor', GradientBoostingClassifier(n_estimators=n_estimators, max_depth=7))
    ])

    esclient = elasticsearch.Elasticsearch()

    y = prime_model(new_source=new_source,
                    new_target=new_target,
                    es_client=esclient,
                    index_name=index_name,
                    doc_type=doc_type,
                    mapping=mapping,
                    usecols=usecols,
                    n_rows_prime=n_rows,
                    scoreplan=scoreplan,
                    n_hits_max=n_hits_max,
                    sbs_score_list=sbs_score_list,
                    pipemodel=pipemodel
                    )
    return y


