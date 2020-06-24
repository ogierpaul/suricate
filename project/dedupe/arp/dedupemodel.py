import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, StandardScaler
from project.dedupe.prime import prime_possible_matches
from pacoetl.utils import extract_dir, pg_engine
from suricate.dbconnectors import EsConnector
from suricate.explore import KBinsCluster
from suricate.sbstransformers import SbsApplyComparator
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def arp_fit_predict(df_source, y_true, es_client, n_hits_max=10, n_estimators=500):
    """

    Args:
        df_source (pd.DataFrame):
        y_true (pd.Series):
        es_client (elasticsearch.Elasticsearch):
        n_hits_max (int):

    Returns:
        pd.DataFrame
    """
    index_name = 'arp'
    usecols = [
        'arp',
        'name',
        'street',
        'city',
        'postalcode',
        'state',
        'countrycode',
        'duns',
        'eu_vat',
        'tax1',
        'tax2',
        'tax3',
        'cage',
        'arp_harmonizedname',
        'arp_partnercompany'
    ]
    doc_type = index_name
    scoreplan = {
        'arp': {'type': 'FreeText'},
        'name': {'type': 'FreeText'},
        'street': {'type': 'FreeText'},
        'city': {'type': 'FreeText'},
        'postalcode': {'type': 'FreeText'},
        'state': {'type': 'Exact'},
        'countrycode': {'type': 'Exact'},
        'duns': {'type': 'Exact'},
        'eu_vat': {'type': 'FreeText'},
        'tax1': {'type': 'FreeText'},
        'tax2': {'type': 'FreeText'},
        'tax3': {'type': 'FreeText'},
        'cage': {'type': 'Exact'},
        'arp_harmonizedname': {'type': 'Exact'},
        'arp_partnercompany': {'type': 'Exact'}
    }
    sbs_score_list = [
        ('name_fuzzy', SbsApplyComparator(on='name', comparator='simple')),
        ('street_fuzzy', SbsApplyComparator(on='street', comparator='simple')),
        ('city_fuzzy', SbsApplyComparator(on='city', comparator='simple')),
        ('postalcode_fuzzy', SbsApplyComparator(on='postalcode', comparator='simple')),
        ('countrycode_exact', SbsApplyComparator(on='countrycode', comparator='exact')),
        ('state_fuzzy', SbsApplyComparator(on='state', comparator='simple')),
        ('duns_exact', SbsApplyComparator(on='duns', comparator='exact')),
        ('cage_exact', SbsApplyComparator(on='cage', comparator='exact')),
        ('arp_harmonizedname_exact', SbsApplyComparator(on='arp_harmonizedname', comparator='exact')),
        ('arp_partnercompany_exact', SbsApplyComparator(on='arp_partnercompany', comparator='exact')),
        ('tax1_fuzzy', SbsApplyComparator(on='tax1', comparator='simple')),
        ('tax2_fuzzy', SbsApplyComparator(on='tax2', comparator='simple')),
        ('tax3_fuzzy', SbsApplyComparator(on='tax3', comparator='simple')),
        ('eu_vat_fuzzy', SbsApplyComparator(on='eu_vat', comparator='simple'))
    ]
    pipemodel = Pipeline(steps=[
        ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaler', Normalizer()),
        ('PCA', PCA(n_components=6)),
        ('Predictor', GradientBoostingClassifier(n_estimators=n_estimators))
    ])
    # Score ES
    Xsm, Xsbs = score_es(df=df_source,
                         esclient=es_client,
                         scoreplan=scoreplan,
                         index_name=index_name,
                         doc_type=doc_type,
                         n_hits_max=n_hits_max,
                         pkey=index_name)

    # Score furthers
    X = score_sbs(df=Xsbs, sbs_score_list=sbs_score_list)
    X = pd.concat([Xsm[['es_score']], X], axis=1, ignore_index=False)

    # Fit the classifier
    pipe = fit_pipeline(pipemodel=pipemodel, X=X, y_true=y_true)

    # Calculate the prediction
    y_pred = pipe.predict(X=X)
    y_proba = pipe.predict_proba(X=X)
    y_proba = pd.DataFrame(y_proba, index=X.index).drop(0, axis=1).rename(columns={1: 'y_proba'})['y_proba']
    df = pd.DataFrame(index=X.index)
    df['y_pred'] = y_pred
    df['y_proba'] = y_proba
    return df


def enrich_output(y_pred, y_proba, Xscores, Xsbs, on_ix=None):
    if on_ix is None:
        on_ix = Xscores.index
    df_res = pd.DataFrame(data=y_proba, columns=[0, 1], index=on_ix)
    df_res.drop(0, axis=1, inplace=True)
    df_res.rename(columns={1: 'y_proba'}, inplace=True)
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

    # Re-order the columns and remove index
    ix_cols = df_res.index.names
    sbscols = list(Xsbs.columns)
    ordercols = ix_cols + sbscols + ['y_pred', 'y_proba', 'cluster_kmeans', 'cluster_kbins']
    missingcols = df_res.columns.difference(set(ordercols))
    ordercols = ordercols + list(missingcols)
    df_res = df_res.reset_index(drop=False)[ordercols]
    return df_res


def fit_pipeline(pipemodel, X, y_true):
    ix_labelled = y_true.index.intersection(X.index)
    X_train = X.loc[ix_labelled]
    y_train = y_true.loc[ix_labelled]
    pipemodel.fit(X=X_train, y=y_train)
    return pipemodel


def score_sbs(df, sbs_score_list):
    scorer_sbs = FeatureUnion(transformer_list=sbs_score_list)
    scores_further = scorer_sbs.fit_transform(X=df)
    scores_further = pd.DataFrame(data=scores_further, index=df.index, columns=[c[0] for c in sbs_score_list])
    return scores_further


def score_es(df, esclient, index_name, doc_type, n_hits_max, scoreplan, pkey):
    escon = EsConnector(
        client=esclient,
        scoreplan=scoreplan,
        index=index_name,
        explain=False,
        size=n_hits_max,
        doc_type=doc_type,
        ixname=pkey
    )
    # Xsm is the similarity matrix
    Xsm = escon.fit_transform(X=df)
    Xsbs = escon.getsbs(X=df, on_ix=Xsm.index)
    return Xsm, Xsbs


def arp_model_kickstartdf(df, nrows=None, n_estimators=500):
    df.index.name = 'ix'
    # prime the model: load from localsuppliers with usecols classics
    possible_pairs = prime_possible_matches(new_source=df, new_target=df, nrows=nrows, n_estimators=n_estimators)
    possible_pairs.index.names = ['arp_source', 'arp_target']
    return possible_pairs


if __name__ == '__main__':
    q_scores = """
    SELECT *
    FROM arp_scores
    LEFT JOIN(SELECT ixp, es_score FROM arp_sbs LIMIT 100) a USING(ixp);
    """
    q_ytrue = """
    SELECT CONCAT(arp_source, '-', arp_target) AS ixp, y_true FROM arp_ytrue
    """
    q_both = """
    SELECT *
    FROM (
             (SELECT CONCAT(arp_source, '-', arp_target) AS ixp, y_true FROM arp_ytrue) y
             LEFT JOIN (SELECT * FROM arp_scores) s USING (ixp)
            LEFT JOIN (SELECT ixp, es_score FROM arp_sbs) a USING (ixp));
"""
    X = pd.read_sql(q_scores, con=pg_engine()).set_index('ixp')
    y_true = pd.read_sql(q_ytrue, con=pg_engine()).set_index('ixp')['y_true']

    print(y_true.value_counts())
    pipemodel = Pipeline(steps=[
        ('Impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('Scaler', StandardScaler())
    ])
    Xt = pipemodel.fit_transform(X)
    Xt = pd.DataFrame(index=X.index, data=Xt)
    X_train = Xt.loc[y_true.index]
    lrc = LogisticRegressionCV(scoring='roc_auc', dual=False, penalty='l2', solver='liblinear')
    gbc = GradientBoostingClassifier(learning_rate=0.005, max_depth=7, max_features='sqrt', n_estimators=600)
    lrc.fit(X_train, y_true)
    gbc.fit(X_train, y_true)
    y_proba = (lrc.predict_proba(Xt) + gbc.predict_proba(Xt)) / 2
    y_proba = y_proba[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    n_bins = 4
    y_proba = pd.Series(y_proba, index=X.index, name='y_proba')
    df = pd.DataFrame(y_proba)
    df['y_pred'] = y_pred
    df.sort_values(by='y_proba', ascending=False)
    df.to_sql(name='arp_pred', if_exists='replace', con=pg_engine())
