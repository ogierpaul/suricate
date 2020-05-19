import pandas as pd
import elasticsearch
# Suricate Model Building
from suricate.data.companies import getsource, gettarget, getytrue
from suricate.dbconnectors import EsConnector
# Sci-kit-learn Model Building
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from suricate.sbstransformers import SbsApplyComparator
from sklearn.pipeline import FeatureUnion

n_rows = 50
n_estimators = 50
n_hits_max = 5
index_name = 'prime'
doc_type = index_name
marker_col = 'origin'

new_source = getsource(nrows=None).tail(n=n_rows)
new_source[marker_col] = 'new'
new_target = gettarget(nrows=None).tail(n=n_rows)
new_target[marker_col] = 'new'

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

usecols = ['name', 'street', 'city', 'postalcode', 'countrycode']


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


esclient = elasticsearch.Elasticsearch()


def check_index_no_overlap(prime_source, new_source, prime_target, primet_target):
    """
    Checkes that the indexes of the prime and new datasets do not have duplicate values
    Args:
        prime_source (pd.DataFrame):
        new_source (pd.DataFrame):
        prime_target (pd.DataFrame):
        primet_target (pd.DataFrame):

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


def score_es(df, esclient, index_name, n_hits_max, scoreplan):
    if scoreplan is None:
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
    escon = EsConnector(
        client=esclient,
        scoreplan=scoreplan,
        index=index_name,
        explain=False,
        size=n_hits_max
    )
    # Xsm is the similarity matrix
    Xsm = escon.fit_transform(X=df)
    Xsbs = escon.getsbs(X=df, on_ix=Xsm.index)
    return Xsm, Xsbs


def score_sbs(df, sbs_score_list):
    if sbs_score_list is None:
        sbs_score_list = [
            ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
            ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
            ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
            ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple'))
        ]
    scorer_sbs = FeatureUnion(transformer_list=sbs_score_list)
    scores_further = scorer_sbs.fit_transform(X=df)
    scores_further = pd.DataFrame(data=scores_further, index=df.index, columns=[c[0] for c in sbs_score_list])
    return scores_further


def fit_pipeline(pipe, X, ix_prime, y_true):
    if pipe is None:
        pipe = Pipeline(steps=[
            ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('Scaler', Normalizer()),
            ('PCA', PCA(n_components=4)),
            ('Predictor', GradientBoostingClassifier(n_estimators=50, max_depth=5))
        ])
    X_prime = X.loc[ix_prime]
    y_prime = y_true.loc[ix_prime]
    pipe.fit(X=X_prime, y=y_prime)
    return pipe


def prime_model(new_source, new_target, es_client, index_name, usecols, n_rows_prime=None, scoreplan=None, n_hits_max=10,
                sbs_score_list=None):
    # Get Priming Data
    prime_source = getsource(nrows=n_rows_prime)
    prime_target = gettarget(nrows=n_rows_prime)
    y_true = getytrue(Xst=[prime_source, prime_target])

    # Concat the two datasets, source and prime
    mix_source = pd.concat([new_source[usecols], prime_source[usecols]], axis=0, ignore_index=False)
    ix_source_prime = prime_source.index
    ix_source_new = new_source.index
    ix_target_prime = prime_target.index
    ix_target_new = new_target.index
    check_index_no_overlap(prime_source, new_source, prime_target, new_target)

    #TODO: Push data to ES

    # Obtain the score from ES (assume both data are in ES)
    Xsm, Xsbs = score_es(df=mix_source, esclient=es_client, scoreplan=scoreplan, index_name=index_name,
                         n_hits_max=n_hits_max)
    ix_con = Xsm.index

    # Obtain the pairs that are from prime datasets and the pairs from new dataset
    ix_prime, ix_new = get_prime_new_index(ix_con, ix_source_prime, ix_source_new, ix_target_prime, ix_target_new,
                                           ixnamesource='ix_source', ixnametarget='ix_target')
    # Score furthers
    X = score_sbs(df=Xsbs, sbs_score_list=sbs_score_list)
    X = pd.concat([Xsm[['es_score']], X], axis=1, ignore_index=False)

    # Fit the pipeline using prime dataset
    pipe = fit_pipeline(X, ix_prime, y_true)

    # Calculate the probability
    y_proba = pipe.predict_proba(X=X.loc[ix_new])
    return y_proba
