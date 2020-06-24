import elasticsearch
import pandas as pd

import pacoetl.createmodel.es
import pacoetl.createmodel.neo
import pacoetl.createmodel.pg
from pacoetl.utils import pg_conn,  staging_dir, neo_bulk_import
from pacoetl.utils import pg_copy_from, es_client, es_index
from pacoetl.utils.others import printmessage

import pacoetl.arp.dbqueries as dbqueries
from pacoetl.arp.transform import clean
from pacoetl.utils.neo import union_cols_neo

global_nrows = None
global_pkey = 'arp'
global_indexname = global_pkey
global_tablename = global_pkey
global_doctype = global_pkey
global_raw_filename = 'arp.csv'


# OK
def read(path, nrows=None):
    """
    Read the csv file at given path for given nrows
    Args:
        path:
        nrows:

    Returns:
        pd.DataFrame
    """
    printmessage('\nstart read {}'.format(path))
    from pacoetl.arp.transform import usecols
    df = pd.read_csv(path, sep='|', nrows=nrows, dtype=str)
    for (k, v) in usecols.to_dict().items():
        if v in df.columns:
            df.rename(columns={v: k}, inplace=True)
    for (k, v) in usecols.to_dict().items():
        if k not in df.columns:
            df[k] = None
    df = df[usecols.index]
    printmessage('end read file with {} number of rows\n'.format(df.shape[0]))
    return df


# OK
def _pg_create():
    conn = pg_conn()
    cur = conn.cursor()
    cur.execute(dbqueries.q_pg_arp_drop)
    cur.execute(pacoetl.createmodel.pg.q_pg_arp_create)
    cur.execute(dbqueries.q_pg_arpytrue_drop)
    cur.execute(dbqueries.q_pg_arpytrue_create)
    cur.execute(dbqueries.q_pg_arphid_drop)
    cur.execute(pacoetl.createmodel.pg.q_pg_arphid_create)
    cur.close()


# OK
def _es_create():
    esclient = elasticsearch.Elasticsearch()
    esclient.indices.delete(index=global_indexname, ignore=[400, 404])
    esclient.indices.create(index=global_indexname, body=pacoetl.createmodel.es.q_es_arp_mapping, ignore=[400, 404])


# OK
def _neo_create(graph):
    graph.run(dbqueries.q_neo_arp_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_arp_create)

    graph.run(dbqueries.q_neo_arppartnercompany_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_arppartnercompany_create)

    graph.run(dbqueries.q_neo_cage_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_cage_create)

    graph.run(dbqueries.q_neo_duns_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_duns_create)

    graph.run(dbqueries.q_neo_suppliername_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_suppliername_create)

    graph.run(dbqueries.q_neo_supplierhid_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_supplierhid_create)

    graph.run(dbqueries.q_neo_tax_drop)
    graph.run(pacoetl.createmodel.neo.q_neo_tax_create)





# OK
def _populate_pg(df, pg_load=True, pg_delete=False):
    conn = pg_conn()
    if pg_delete is True:
        cur = conn.cursor()
        cur.execute('DELETE FROM arp')
        cur.close()
    if pg_load is True:
        pg_copy_from(df, conn, global_pkey, staging_dir, global_pkey)
    return True


# OK
def _populate_es(df, es_load=True, es_delete=False):
    client = es_client()
    if es_delete is True:
        client.delete_by_query(index=global_indexname, body={"query": {"match_all": {}}})
    if es_load is True:
        es_index(client, df, global_indexname, global_doctype, global_pkey, sleep=3)
    return True


# OK
def _populate_neo(df, graph, neo4j_dir):
    """
    Will MERGE (~ Upsert) neo from a well-formatted DataFrame containing the ARP Data
    - ARP Nodes
    - Cage Nodes
    - Duns Nodes
    - ArpPartnerCompany
    - ArpHarmonizedName and ARP Supplier Name
    - TaxId: Eu_vat, Tax1,2,3
    Args:
        df (pd.DataFrame): index is 'arp'
        graph (py2neo): py2neo connector
        neo4j_dir (str): path to the neo4j import dir for Bulk Import using LOAD CSV

    Returns:
        None
    """
    # Load ARP
    neo_bulk_import(df=df, importdir=neo4j_dir, graph=graph,
                    fileprefix='arp', query=dbqueries.q_neo_arp_load)
    printmessage('Neo ARP Nodes')
    # Load ARp to Cage relationships
    neo_bulk_import(df=df[['cage']].dropna(), importdir=neo4j_dir, graph=graph,
                    fileprefix='cage', query=dbqueries.q_neo_cage_load)
    printmessage('Neo CAGE Nodes')
    # Load ARP to Duns Relationships
    neo_bulk_import(df=df[['duns']].dropna(), importdir=neo4j_dir, graph=graph,
                    fileprefix='duns', query=dbqueries.q_neo_duns_load)
    printmessage('Neo Duns Nodes')
    # Load Arp to Arp PartnerCompany relationships
    neo_bulk_import(df=df[['arp_partnercompany']].dropna(), importdir=neo4j_dir, graph=graph,
                    fileprefix='arp_partnercompany', query=dbqueries.q_neo_arppartnercompany_load)
    printmessage('Neo Arp_PartnerCompany Nodes')
    # Load Arp to Taxid relationship
    tax = union_cols_neo(df, pkey='arp', cols=['eu_vat', 'tax1', 'tax2', 'tax3'])
    neo_bulk_import(df=tax, importdir=neo4j_dir, graph=graph,
                    fileprefix='tax', query=dbqueries.q_neo_tax_load)
    printmessage('Neo Tax Nodes')
    # Load Arp to Name relationships
    arpname = df[['name']].dropna().reset_index(drop=False).rename(columns={'name': 'suppliername'})
    arpname['type'] = 'arpname'
    arp_harmonizedname = df[['arp_harmonizedname']].dropna().reset_index(drop=False).rename(
        columns={'arp_harmonizedname': 'suppliername'})
    arp_harmonizedname['type'] = 'arp_harmonizedname'
    suppliername = pd.concat([arpname, arp_harmonizedname], axis=0, ignore_index=True).set_index('arp')
    neo_bulk_import(df=suppliername, importdir=neo4j_dir, graph=graph,
                    fileprefix='suppliername', query=dbqueries.q_neo_suppliername_load)
    printmessage('Neo SupplierName Nodes')


def _check_data(n_records, graph):
    conn = pg_conn()
    cur = conn.cursor()
    select_sql = """
    SELECT COUNT(*) FROM arp;
    """
    cur.execute(select_sql)
    r = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    pg_records = r[0]

    es_records = es_client().count(doc_type=global_doctype, index=global_indexname)['count']

    match_neo = """
    MATCH (n:Arp)
    RETURN COUNT(n) as count
    """
    neo_records = graph.run(match_neo).data()[0]['count']

    assert n_records == pg_records
    assert n_records == es_records
    assert n_records == neo_records
    return True


# # OK
# def populate_bws():
#     printmessage(' | Start populating table arp')
#     df = read(path=extract_dir + global_raw_filename, nrows=global_nrows)
#     printmessage('Read successfull with {} number of lines\n'.format(df.shape[0]))
#     df = clean(df)
#     n_records = df.shape[0]
#     printmessage('Clean successfull with {} number of records\n'.format(n_records))
#     _populate_pg(df)
#     printmessage('PostgreSQL Load successfull\n')
#     _populate_es(df)
#     printmessage('ElasticSearch Load successfull\n')
#     _populate_neo(df)
#     printmessage('Neo4j Load successfull\n')
#     _check_data(n_records)
#     printmessage('Consistent Number of records: {}\n'.format(n_records))
#     return True

# OK
def arp_neo_read_clean_load(path, graph, import_dir):
    """
    Read, clean, and load into neo
    Args:
        path (str): Path to arp raw filename
        graph (py2neo.Graph): py2neo connector instance
        import_dir: neo4j Import Dir

    Returns:
        None
    """
    df = read(path=path)
    df = clean(df)
    _populate_neo(df, graph, import_dir)
    return None




