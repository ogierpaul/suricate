import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
import elasticsearch
from suricate.dbconnectors import EsConnector
from pacoetl.utils import es_create_load
from suricate.explore import KBinsCluster
from suricate.sbstransformers import SbsApplyComparator



def companies_fit_predict(df_source, df_target, y_true, es_client, index_name, n_estimators=500, doc_type=None, mapping=None, usecols=None,
                          scoreplan=None, pipemodel=None, n_hits_max=10,
                          sbs_score_list=None, pkey='ix'):
    """

    Args:
        df_source:
        df_target:
        y_true:
        es_client:
        index_name:
        doc_type:
        mapping:
        usecols:
        scoreplan:
        pipemodel:
        n_hits_max:
        sbs_score_list:
        pkey:

    Returns:
        pd.DataFrame: with index (ixnamesource, ixnametarget)
    """
    if usecols is None:
        usecols = ['name', 'street', 'city', 'postalcode', 'countrycode']
    if doc_type is None:
        doc_type = index_name
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
        'postalcode': {
            'type': 'FreeText'
        },
        'countrycode': {
            'type': 'Exact'
        }
    }
    if mapping is None:
        mapping = {
        "mappings": {
            doc_type: {
                "properties": {
                    pkey: {"type": "keyword"},
                    "name": {"type": "text"},
                    "street": {"type": "text"},
                    "city": {"type": "text"},
                    "postalcode": {"type": "text"},
                    "countrycode": {"type": "keyword"}
                }
            }
        }
    }
    if sbs_score_list is None:
        sbs_score_list = [
                ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
                ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
                ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
                ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple'))
            ]
    if pipemodel is None:
        pipemodel = Pipeline(steps=[
            ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('Scaler', Normalizer()),
            ('PCA', PCA(n_components=3)),
            ('Predictor', GradientBoostingClassifier(n_estimators=n_estimators, max_depth=7))
        ])
    if es_client is None:
        es_client = elasticsearch.Elasticsearch()
    print(pd.datetime.now(), 'start data loading')
    es_create_load(df=df_target, client=es_client, index=index_name, mapping=mapping, doc_type=doc_type, index_id=pkey,
                   drop=True, create=True)
    print(pd.datetime.now(), 'data loaded in es')
    Xsm, Xsbs = score_es(df=df_source, client=es_client, scoreplan=scoreplan, index=index_name,
                         doc_type=doc_type,
                         n_hits_max=n_hits_max, id=pkey)
    print(pd.datetime.now(), 'data scored from es')
    # Score furthers
    X = score_sbs(df=Xsbs, sbs_score_list=sbs_score_list)
    X = pd.concat([Xsm[['es_score']], X], axis=1, ignore_index=False)
    print(pd.datetime.now(), 'data scored further')
    # Fit the classifier
    pipe = fit_pipeline(pipemodel=pipemodel, X=X, y_true=y_true)
    print(pd.datetime.now(), 'pipe fitted')
    # Calculate the prediction
    y_pred = pipe.predict(X=X)
    y_proba = pipe.predict_proba(X=X)
    print(pd.datetime.now(), 'pipe predicted')
    df_res = enrich_output(y_pred=y_pred, y_proba=y_proba, Xscores=X,  Xsbs=Xsbs)
    return df_res


def enrich_output(y_pred, y_proba, Xscores, Xsbs, on_ix=None):
    if on_ix is None:
        on_ix = Xscores.index
    df_res = pd.DataFrame(data=y_proba, columns=[0, 1], index=on_ix)
    df_res.drop(0, axis=1, inplace=True)
    df_res.rename(columns={1:'y_proba'}, inplace=True)
    df_res['y_pred'] = y_pred

    # Add some scores
    df_res = pd.concat([df_res, Xscores.loc[on_ix]], axis=1, ignore_index=False)

    # Enrich the data with clusters
    km = KMeans(n_clusters=10)
    km.fit(X=Xscores)
    y_km = km.predict(X=Xscores.loc[on_ix])
    df_res['cluster_kmeans'] = y_km

    # Enrich the data with kbins discretizer
    kb = KBinsCluster(n_clusters=10, encode='ordinal', strategy='quantile')
    kb.fit(X=Xscores)
    y_kb = kb.transform(X=Xscores)
    y_kb = pd.DataFrame(data=y_kb, index=Xscores.index, columns=['cluster_kbins'])
    y_kb = y_kb.loc[on_ix]
    y_kb = y_kb[['cluster_kbins']]
    df_res['cluster_kbins'] = y_kb

    # Add Side-by-side comparison
    df_res = pd.concat([df_res, Xsbs.loc[on_ix]], axis=1, ignore_index=False)

    #Re-order the columns and remove index
    ix_cols = df_res.index.names
    sbscols = list(Xsbs.columns)
    ordercols = sbscols + ['y_pred', 'y_proba', 'cluster_kmeans', 'cluster_kbins']
    missingcols = df_res.columns.difference(set(ordercols))
    ordercols = ordercols + list(missingcols)
    df_res = df_res[ordercols]
    return df_res


def fit_pipeline(pipemodel, X, y_true):
    ix_labelled= y_true.index.intersection(X.index)
    X_train = X.loc[ix_labelled]
    y_train = y_true.loc[ix_labelled]
    pipemodel.fit(X=X_train, y=y_train)
    return pipemodel


def score_sbs(df, sbs_score_list):
    scorer_sbs = FeatureUnion(transformer_list=sbs_score_list)
    scores_further = scorer_sbs.fit_transform(X=df)
    scores_further = pd.DataFrame(data=scores_further, index=df.index, columns=[c[0] for c in sbs_score_list])
    return scores_further


def score_es(df, client, index, doc_type, n_hits_max, scoreplan, id='ix'):
    escon = EsConnector(
        client=client,
        scoreplan=scoreplan,
        index=index,
        explain=False,
        size=n_hits_max,
        doc_type=doc_type,
        ixname=id
    )
    # Xsm is the similarity matrix
    Xsm = escon.fit_transform(X=df)
    Xsbs = escon.getsbs(X=df, on_ix=Xsm.index)
    return Xsm, Xsbs