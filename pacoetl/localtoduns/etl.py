import pandas as pd
import datetime

import pacoetl.createmodel.neo
import pacoetl.createmodel.pg
from pacoetl.utils import pg_conn, pg_copy_from, staging_dir, neo_bulk_import
from pacoetl.utils.clean import clean_inputs, format_duns, format_arp
import pacoetl.localtoduns.dbqueries


global_arp_system_list = ('X11', 'XGI', 'XEA', 'AP1', 'APD', 'SPA', 'XDA', 'ARP', 'PBA')


def create():
    con = pg_conn()
    cur = con.cursor()
    cur.execute(pacoetl.localtoduns.dbqueries.q_pg_localidduns_drop)
    cur.execute(pacoetl.createmodel.pg.q_pg_localidduns_create)
    cur.close()


def read(path):
    df = pd.read_csv(path, dtype=str, sep='|', nrows=None)
    return df


def clean(df):
    usecols = pd.Series({
        'supplier_id': 'localid',
        'source_system': 'sysid',
        'duns': 'old_duns',
        'newduns': 'new_duns',
    })
    df = df.loc[df['source_system'].isin(global_arp_system_list)]
    df = clean_inputs(df=df, usecols=usecols)
    df = format_duns(df, col='old_duns')
    df = format_duns(df, col='new_duns')
    df['duns'] = df['old_duns'].combine_first(df['new_duns'])
    df = df[['sysid', 'localid', 'old_duns', 'new_duns', 'duns']]
    df = df.dropna(subset=['duns']).drop_duplicates(subset=['duns']).set_index(['sysid', 'localid'])
    return df


def _populate_pg(df):
    con = pg_conn()
    pg_copy_from(df, con, 'localsuppliers_duns', staging_dir=staging_dir, pkey=['sysid', 'localid'])


def _populate_neo(df, graph, neo4j_dir):
    df = df[['duns']].reset_index()
    df = format_arp(df, col='localid')
    df = df.set_index(['sysid', 'localid'])
    neo_bulk_import(df=df, importdir=neo4j_dir, graph=graph,
                    fileprefix='arpduns', query=pacoetl.localtoduns.dbqueries.q_neo_arpduns_load)
    print(datetime.datetime.now(), ' | Neo4j Arp 2 Duns Load successfull\n')

#
# def populate_bw():
#     print(datetime.datetime.now(), ' | Start populating table localsuppliers to duns')
#     df = read()
#     print(datetime.datetime.now(), ' | Read successfull with {} number of lines\n'.format(df.shape[0]))
#
#     df = clean(df)
#     print(df.head())
#     print(datetime.datetime.now(), ' | Clean successfull with {} number of records\n'.format(df.shape[0]))
#
#     _populate_pg(df)
#     print(datetime.datetime.now(), ' | PostgreSQL Load successfull\n')
#
#     _populate_neo(df)
#     print(datetime.datetime.now(), ' | Neo4j Load successfull\n')


def localtoduns_neo_read_clean_load(path, graph, import_dir):
    """
    Read, clean, and load into neo
    Args:
        path (str): Path to arp raw filename
        graph (py2neo.Graph): py2neo connector instance
        import_dir: neo4j Import Dir

    Returns:
        None
    """
    df = read(path)
    df = clean(df)
    _populate_neo(df, graph, import_dir)
    return None
