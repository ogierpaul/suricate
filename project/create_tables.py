from psycopg2 import sql
from pacoetl.utils import pg_conn, neo_graph
import elasticsearch

pg_create_supplierharmonized = sql.SQL("""
CREATE TABLE IF NOT EXISTS supplierharmonized (
    hid VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP,
    PRIMARY KEY (hid));
""")
pg_delete_supplierharmonized = sql.SQL(""" DROP TABLE IF EXISTS supplierharmonized;""")


# EPROC
eproc_pg_create = sql.SQL("""
CREATE TABLE IF NOT EXISTS eproc (
    ariba VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    arp VARCHAR,
    duns VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP,
    PRIMARY KEY (ariba));
""")
eproc_pg_drop= sql.SQL(""" DROP TABLE IF EXISTS eproc;""")
eproc_neo_create = """
CREATE CONSTRAINT ON (n:Eproc) ASSERT n.ariba IS UNIQUE
"""
eproc_neo_delete = """
MATCH (n:Eproc)
DETACH DELETE n;
"""

create_duns = sql.SQL("""
CREATE TABLE IF NOT EXISTS duns (
    duns VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP,
    PRIMARY KEY (duns));
""")
delete_duns = sql.SQL(""" DROP TABLE IF EXISTS duns;""")

localsuppliers_pg_create = sql.SQL("""
CREATE TABLE IF NOT EXISTS localsuppliers (
    uid VARCHAR,
    sysid VARCHAR,
    localid VARCHAR,
    hid VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    arp VARCHAR,
    duns VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP,
    PRIMARY KEY (uid));
""")
localsuppliers_pg_drop = sql.SQL(""" DROP TABLE IF EXISTS localsuppliers;""")
localsuppliers_neo_create = """
CREATE CONSTRAINT ON (n:LocalSupplier) ASSERT n.uid IS UNIQUE
"""
localsuppliers_neo_delete = """
MATCH (n:LocalSupplier)
DETACH DELETE n;
"""

localsuppliers_mapping = {
    "mappings": {
        "localsuppliers": {
            "properties": {
                "uid": {"type": "keyword"},
                "sysid": {"type": "keyword"},
                "localid": {"type": "keyword"},
                "hid": {"type": "keyword"},
                "name": {"type": "text"},
                "street": {"type": "text"},
                "city": {"type": "text"},
                "postalcode": {"type": "keyword"},
                "state": {"type": "text"},
                "countrycode": {"type": "keyword"},
                "arp": {"type": "keyword"},
                "duns": {"type": "keyword"},
                "eu_vat": {"type": "text"},
                "tax1": {"type": "text"},
                "tax2": {"type": "text"},
                "tax3": {"type": "text"},
                "extract_ts": {"type": "date"}
            }
        }
    }
}


def create_eproc():
    conn = pg_conn()
    esclient = elasticsearch.Elasticsearch()
    neograph = neo_graph()
    tablename = 'eproc'
    index = tablename
    cur = conn.cursor()
    cur.execute(eproc_pg_drop)
    cur.execute(eproc_pg_create)
    neograph.run(eproc_neo_delete)
    neograph.run(eproc_neo_create)
    pass

def create_localsuppliers():
    conn = pg_conn()
    esclient = elasticsearch.Elasticsearch()
    neograph = neo_graph()
    tablename = 'localsuppliers'
    index = tablename
    cur = conn.cursor()
    cur.execute(localsuppliers_pg_drop)
    cur.execute(localsuppliers_pg_create)
    neograph.run(localsuppliers_neo_delete)
    neograph.run(localsuppliers_neo_create)
    esclient.indices.delete(index=index, ignore=[400, 404])
    esclient.indices.create(index=index, body=localsuppliers_mapping, ignore=[400, 404])
    pass



