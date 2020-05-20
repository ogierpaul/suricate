import pandas as pd
from paco.utils import clean_inputs, pg_conn, copy_from, es_index
import elasticsearch
from psycopg2 import sql
import datetime

tablename = 'arp'
pkey = tablename
indexname = tablename
doc_type = tablename
extract_path = '../../tests/extract_dir/arp.csv'
staging_dir = '../../tests/staging'
usecols = pd.Series(

    data=['arp', 'name', 'street', 'city', 'postalcode', 'state', 'countrycode', 'iban', 'duns', 'ssn']
)
coltypes = {
    'arp': 'str',
    'postalcode': 'str',
    'duns': 'str'
}
colzeroes = {
    'arp': 6,
    'duns': 9
}
nrows = 100

mymapping = {
    "mappings": {
        doc_type: {
            "properties": {
                "arp": {"type": "keyword"},
                "name": {"type": "text"},
                "street": {"type": "text"},
                "city": {"type": "text"},
                "postalcode": {"type": "keyword"},
                "state": {"type": "text"},
                "countrycode": {"type": "keyword"},
                "iban": {"type": "keyword"},
                "duns": {"type": "keyword"},
                "ssn": {"type": "keyword"},
                "extract_ts": {"type": "date"}
            }
        }
    }
}


def clean_func(df):
    """

    Args:
        df (pd.DataFrame)
    Returns:
        pd.DataFrame
    """
    df2 = df.loc[df['countrycode'] != 'ZA']
    df2['duns'] = df2['duns'].str.replace('-', '')
    df2['extract_ts'] = datetime.datetime.now()
    return df2


def pg_create_table(conn, tablename):
    # noinspection SyntaxError
    create_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {tablename} (
        arp VARCHAR,
        name VARCHAR,
        street VARCHAR,
        city VARCHAR,
        postalcode VARCHAR ,
        state VARCHAR ,
        countrycode VARCHAR,
        iban VARCHAR ,
        duns VARCHAR,
        ssn VARCHAR,
        extract_ts TIMESTAMP,
        PRIMARY KEY (arp)
    );
    """).format(tablename=sql.Identifier(tablename))
    cur = conn.cursor()
    cur.execute(create_sql)
    conn.commit()
    cur.close()
    return True


def pg_model(df):
    """
    Model the data for ingestion into Postgres
    Do nothing
    Args:
        df (pd.DataFrame):

    Returns:
        pd.DataFrame
    """
    return df


def pg_create_load(df, conn, drop, create, tablename, staging_dir, pkey):
    """
    Create the table and load the data into it
    Args:
        df(pd.DataFrame): Data
        conn (psycopg2.connection): Connector
        drop (bool): If True, proceed first to drop the table
        create (bool): If True, proceed then to create the table
        tablename (str): Name of the table in Postgres
        staging_dir (str): Path of the dir where to write the csv for COPY FROM
        pkey (str/list): Primary key in Postgres

    Returns:
        pd.DataFrame
    """
    cur = conn.cursor()
    if drop is True:
        for t in tablename, 'temp_' + tablename:
            cur.execute(sql.SQL("""DROP TABLE IF EXISTS {tablename};""").format(tablename=sql.Identifier(t)))
        conn.commit()
        cur.close()
    if create is True:
        pg_create_table(conn=conn, tablename=tablename)
    copy_from(df=df, conn=conn, tablename=tablename, staging_dir=staging_dir, pkey=pkey)
    return None


def es_create_load(df, es_client, drop, create, indexname, doc_type, pkey, mapping):
    """
    Create the index & load the data into the index
    Args:
        df(pd.DataFrame):
        es_client (elasticsearch.Elasticsearch): elastic search client
        drop (bool): If True, proceed first to drop the index
        create (bool): If True, proceed then to create the index
        indexname (str): Name of index in ElasticSearch. (also called indice).
        doc_type (str): Name of document type in ElasticSearch. If None, uses the same name as the indexname
        mapping (dict): Mapping of the index
        pkey (str): String

    Returns:
        None
    """
    if doc_type is None:
        doc_type = indexname
    if drop is True:
        es_client.indices.delete(index=indexname, ignore=404)
    if create is True:
        es_client.indices.create(index=indexname, body=mapping, ignore=400)
    es_index(client=es_client, df=df, index=indexname, doc_type=doc_type, id=pkey, sleep=2)
    return None


def es_model(df):
    """
    Model the data for ingestion into ElasticSearch
    Change Timestamp format
    Args:
        df (pd.DataFrame):

    Returns:
        pd.DataFrame
    """
    df2 = df
    timecol = 'extract_ts'
    df2[timecol] = df[timecol].apply(lambda r: r.isoformat())
    return df


def main():
    drop_pg = True
    create_pg = True
    drop_es = True
    create_es = True

    # Extract
    df = pd.read_csv(extract_path, sep=',', nrows=nrows, dtype=str)
    print(datetime.datetime.now(), ' | Read successfull')

    # Clean
    df2 = clean_inputs(df=df, pkey=pkey, usecols=usecols, sep_csv='|', coltypes=coltypes, colzeroes=colzeroes,
                       transform_func=clean_func)
    print(datetime.datetime.now(), ' | Clean successfull')

    # LOAD INTO PG
    conn = pg_conn()
    df_pg = pg_model(df2)
    pg_create_load(df=df_pg, conn=conn, drop=drop_pg, create=create_pg, tablename=tablename, staging_dir=staging_dir,
                   pkey=pkey)
    print(datetime.datetime.now(), ' | load into PG successfull')

    # LOAD INTO ES
    e = elasticsearch.Elasticsearch()
    df_es = es_model(df2)
    es_create_load(df=df_es, es_client=e, drop=drop_es, create=create_es, indexname=tablename, doc_type=doc_type, pkey=pkey, mapping=mymapping)
    print(datetime.datetime.now(), ' | load into ES successfull')


if __name__ == '__main__':
    main()
