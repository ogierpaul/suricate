import datetime

import numpy as np
import pandas as pd

from pacoetl.arp import dbqueries as dbqueries
from pacoetl.arp.etl import global_nrows
from pacoetl.utils import clean_inputs, pg_engine, extract_dir, pg_conn, es_client, neo_graph, deduperels, neo_bulk_import, \
    neo4j_dir
from pacoetl.utils.clean import rmv_blank_values, format_duns, concat_prefix, idtostr
from pacoetl.utils.others import uniquepairs
from project.dedupe.arp.dedupemodel import arp_fit_predict, arp_model_kickstartdf

df = pd.read_sql('SELECT * from arpdedupe.arp_ytrue', con=pg_engine())
df['ixp'] = df[['arp_source', 'arp_target']].apply(lambda r:uniquepairs(r['arp_source'], r['arp_target']), axis=1)
df = df[['ixp', 'arp_source', 'arp_target', 'y_true']]
df.to_sql('arp_ytrue', schema='arpdedupe', con=pg_engine(), index=False, if_exists='replace')


def update_y_true():
    df = pd.read_csv(extract_dir + 'arp_ytrue.csv', sep='|')
    df = df.loc[~df['y_true'].isnull()]
    df['arp_source'] = df['ixp'].str.split('-').str[0]
    df['arp_target'] = df['ixp'].str.split('-').str[1]
    df['y_true'] = df['y_true'].astype(int)
    y_truedf = df[['arp_source', 'arp_target', 'y_true']].dropna()
    y_trueseries = y_truedf.set_index(['arp_source', 'arp_target'])['y_true']
    cur = pg_conn().cursor()
    for (i, r) in y_truedf.iterrows():
        cur.execute(dbqueries.q_pg_arpytrue_update, r)
    cur.close()
    # _neo_updateytrue(y_trueseries)
    print(datetime.datetime.now(), ' | arp_ytrue loaded')
    return True


def arp_model_fitpred():
    df_source = pd.read_sql(dbqueries.q_pg_arp_select, con=pg_engine()).set_index(['arp'])
    y_true = \
        pd.read_sql(dbqueries.q_pg_arpytrue_select, con=pg_engine()).set_index(
            ['arp_source', 'arp_target'])[
            'y_true']
    print(datetime.datetime.now(), ' | df and y_true read')
    print(y_true.value_counts())
    results = arp_fit_predict(df_source, y_true, es_client=es_client(), n_estimators=global_nestimators)
    print(datetime.datetime.now(), ' | results pred')
    results.to_sql(name='arp_possiblepairs', con=pg_engine(), index=True, if_exists='replace')
    print(datetime.datetime.now(), ' | results stored')
    return True


def _neo_updateytrue2():
    """

    Args:

    Returns:
        bool
    """
    df = pd.read_sql('SELECT arp_source, arp_target, y_pred FROM arp_pred', con=pg_engine())
    y = df.set_index(['arp_source', 'arp_target'])['y_pred']
    y = y.loc[y==1]
    graph = neo_graph()
    graph.run(dbqueries.q_neo_arpytrue_delete)
    z = deduperels(y, aggfunc=np.nanmax)
    z = pd.DataFrame(z)
    neo_bulk_import(z, importdir=neo4j_dir, graph=graph,
                    fileprefix='arp_ypred', query=dbqueries.q_neo_arpytrue_load)


def prime_possiblepairs():
    df = pd.read_sql(dbqueries.q_pg_arp_select, con=pg_engine()).set_index(['arp'])
    print(df.shape[0])
    possible_pairs = arp_model_kickstartdf(df, nrows=global_nrows, n_estimators=global_nestimators)
    print(datetime.datetime.now(), ' | Possible pairs calculated')
    possible_pairs.to_sql(name='arp_possiblepairs', con=pg_engine(), index=True, if_exists='replace')
    _neo_updateyproba(possible_pairs)
    print(datetime.datetime.now(), ' | Possible pairs loaded')
    return True


def _neo_updateyproba(df):
    """

    Args:
        df (pd.DataFrame): with multiindex and y_proba

    Returns:
        bool
    """
    n_bins = 3
    graph = neo_graph()
    graph.run(dbqueries.q_neo_possiblepairs_delete)
    z = deduperels(df['y_proba'])
    z = pd.DataFrame(z)
    z['y_discrete'] = pd.cut(z['y_proba'], bins=n_bins, labels=range(n_bins))
    neo_bulk_import(z, importdir=neo4j_dir, graph=graph,
                    fileprefix='possiblepairs', query=dbqueries.q_neo_possiblepairs_load)
    return True


def _neo_updateytrue(y):
    """

    Args:
        y (pd.Series): with multiindex and y_proba

    Returns:
        bool
    """
    graph = neo_graph()
    graph.run(dbqueries.q_neo_arpytrue_delete)
    z = deduperels(y, aggfunc=np.nanmax)
    z = pd.DataFrame(z)
    neo_bulk_import(z, importdir=neo4j_dir, graph=graph,
                    fileprefix='arp_ytrue', query=dbqueries.q_neo_arpytrue_load)


global_nestimators = 500