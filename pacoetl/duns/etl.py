import pandas as pd


import pacoetl.createmodel.es
import pacoetl.createmodel.pg
import pacoetl.createmodel.neo
from pacoetl.utils import pg_conn, pg_copy_from, staging_dir, neo_bulk_import, es_client, es_index
from pacoetl.utils.clean import clean_inputs, format_duns, concat_cols, format_tax
from pacoetl.utils.others import printmessage
import pacoetl.duns.dbqueries as dbqueries

global_pkey ='duns'
global_indexname = global_pkey
global_doctype = global_pkey

def create(graph):
    #PG
    con = pg_conn()
    cur = con.cursor()
    cur.execute(dbqueries.q_pg_duns_drop)
    cur.execute(pacoetl.createmodel.pg.q_pg_duns_create)
    cur.close()

    #NEO
    graph.run(pacoetl.createmodel.neo.q_neo_duns_create)

    #ES
    esclient = es_client()
    esclient.indices.delete(index=global_indexname, ignore=[400, 404])
    esclient.indices.create(index=global_indexname, body=pacoetl.createmodel.es.q_es_duns_mapping, ignore=[400, 404])


def read(path, nrows=None):
    printmessage('\nstart read {}'.format(path))
    df = pd.read_csv(path, dtype=str, sep='|', nrows=nrows)
    printmessage('end read file with {} number of rows'.format(df.shape[0]))
    return df


def clean(df):
    printmessage('start cleaning file')
    usecols = pd.Series({
        'dunsno': 'new_duns',
        'name': 'name',
        '2nd_name': 'name2',
        'add': 'street',
        'add2': 'street2',
        'city': 'city',
        'state': 'state',
        'postcode': 'postalcode',
        'countrycode': 'countrycode',
        'natid': 'tax1',
        'statuscode': 'statuscode',
        'subscode': 'subscode',
        'prevduns': 'old_duns',
        'userarea': 'userarea'
    })
    ordercols = [
        'duns',
        'name',
        'street',
        'postalcode',
        'city',
        'state',
        'countrycode',
        'tax1',
        'statuscode',
        'subscode',
        'userarea'
    ]
    df = clean_inputs(df=df, usecols=usecols, clean_js=False, clean_csv=False)
    df = format_duns(df, col='old_duns')
    df = format_duns(df, col='new_duns')
    df = concat_cols(df, 'name', ['name', 'name2'])
    df = concat_cols(df, 'street', ['street', 'street2'])
    df['duns'] = df['old_duns'].combine_first(df['new_duns'])
    df = format_tax(df, 'tax1', 'countrycode')
    df.drop_duplicates(subset=['duns'], inplace=True)
    df = df[ordercols]
    df.set_index(['duns'], inplace=True)
    printmessage('end cleaning file')
    return df


def _populate_pg(df):
    con = pg_conn()
    pg_copy_from(df, con, 'duns', staging_dir=staging_dir, pkey=['duns'])

def _populate_neo(df, graph, neo4j_dir):
    neo_bulk_import(df=df, importdir=neo4j_dir, graph=graph,
                    fileprefix='duns', query=pacoetl.duns.dbqueries.q_neo_duns_load)
    printmessage('Neo Duns Nodes')
    tax = df[['tax1']].dropna().rename(columns={'tax1':'uid'})
    tax['type'] = 'natid'
    neo_bulk_import(df=tax, importdir=neo4j_dir, graph=graph,
                    fileprefix='tax', query=pacoetl.duns.dbqueries.q_neo_tax_load)
    printmessage('Neo Tax Nodes')
    suppliername = df[['name']].dropna().rename(columns={'name': 'suppliername'})
    neo_bulk_import(df=suppliername, importdir=neo4j_dir, graph=graph,
                    fileprefix='tax', query=pacoetl.duns.dbqueries.q_neo_suppliername_load)
    printmessage('Neo SupplierName Nodes')
    return True

# OK
def _populate_es(df):
    es_delete = False
    es_load = True
    client = es_client()
    if es_delete is True:
        client.delete_by_query(index=global_indexname, body={"query": {"match_all": {}}})
    if es_load is True:
        es_index(client, df, global_indexname, global_doctype, global_pkey, sleep=3)
    return True


# def populate_bw():
#     printmessage('Start populating table duns')
#     df = read()
#     printmessage('Read successfull with {} number of lines\n'.format(df.shape[0]))
#
#     df = clean(df)
#     printmessage('Clean successfull with {} number of records\n'.format(df.shape[0]))
#
#     _populate_pg(df)
#     printmessage('PostgreSQL Load successfull\n')
#
#     _populate_es(df)
#     printmessage('ElasticSearch Load successfull\n')
#
#     _populate_neo(df)
#     printmessage('Neo4j Load successfull\n')


def duns_neo_read_clean_load(path, graph, import_dir):
    """
    Read, clean, and load into neo
    Args:
        path (str): Path to duns raw file
        graph (py2neo.Graph): py2neo connector instance
        import_dir: neo4j Import Dir

    Returns:
        None
    """
    df = read(path)
    df = clean(df)
    _populate_neo(df, graph, import_dir)
    return None
