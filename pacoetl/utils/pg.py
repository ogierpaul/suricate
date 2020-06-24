import os
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql

from pacoetl.utils.others import write_csv


def _format_pkey(pkey):
    """
    Format the primary key to be inserted into an SQL query
    Args:
        pkey (str/list):

    Returns:
        str: pkey: column name or list of column names separated by commar
    """
    if isinstance(pkey, str):
        pkey_s = pkey
    else:
        pkey_s = ', '.join(pkey)
    return pkey_s


def copy_wopkey(tablename, filepath, cur, sep='|'):
    """
    copy_wopkey
    Args:
        tablename (str):
        filepath (str):
        cur (psycopg2.cursor):
        sep (str): separator

    Returns:
        None
    """
    # Preventing SQL injections thanks to https://github.com/psycopg/psycopg2/issues/529
    query = sql.SQL("""
        COPY {tablename} FROM STDIN WITH CSV HEADER ENCODING 'UTF-8' DELIMITER '{sep}'
        """).format(tablename=sql.Identifier(tablename), sep=sql.Literal(sep))
    with open(filepath, 'r') as f:
        cur.copy_expert(query, f)
    return None


# noinspection SyntaxError
def copy_withpkey(tablename, filepath, cur, sep, pkey):
    """
    copy from filepath to tablename with delimiter sep
        - Create a temporary empty temp_tablename with same structure as tablename
        - COPY FROM the input data to a temp_tablename
        - INSERT / ON CONFLICT DO NOTHING between temp_tablename and tablename
        - DROP temp_tablename
    Args:
        tablename (str):
        filepath (str):
        cur (psycopg2.cursor):
        sep (str): separator
        pkey (str/list):

    Returns:
        None
    """
    # COPIED FROM https://www.postgresql.org/message-id/464F7A31.6020501@autoledgers.com.au
    # Preventing SQL injections thanks to https://realpython.com/prevent-python-sql-injection/
    temp_tablename = 'temp_' + tablename
    # Create temp_tablename
    query_create = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {temp_tablename} AS SELECT * FROM {tablename} WHERE 1=0;
    TRUNCATE TABLE {temp_tablename};""").format(
        temp_tablename=sql.Identifier(temp_tablename),
        tablename=sql.Identifier(tablename)
    )
    cur.execute(query_create)
    # Delete rows from temp_tablename
    # noinspection SyntaxError
    query_delete = sql.SQL("""DELETE FROM {temp_tablename};""").format(temp_tablename=sql.Identifier(temp_tablename))
    cur.execute(query_delete)
    # Copy from CSV to temp_tablename
    query_copy = sql.SQL("""
    COPY {temp_tablename} FROM STDIN WITH 
    DELIMITER AS '|' ENCODING 'UTF-8' CSV HEADER;
    """).format(temp_tablename=sql.Identifier('temp_' + tablename))
    with open(filepath, 'r') as f:
        cur.copy_expert(query_copy, f)
    if isinstance(pkey, str):
        pkey_s = sql.Identifier(pkey)
    else:
        pkey_s = sql.SQL(', ').join([sql.Identifier(c) for c in pkey])
    # Upsert from temp_tablename into tablename
    query_upsert = sql.SQL("""
    INSERT INTO {tablename}
        (
            SELECT DISTINCT ON ({pkey_s}) *
            FROM {temp_tablename}
            WHERE ({pkey_s}) is not null
        )
    ON CONFLICT ({pkey_s})
    DO NOTHING;
    """).format(tablename=sql.Identifier(tablename),
                temp_tablename=sql.Identifier(temp_tablename),
                pkey_s=pkey_s)
    cur.execute(query_upsert)
    # Drop temp_tablename
    query_drop = sql.SQL("""DROP TABLE {temp_tablename};""").format(temp_tablename=sql.Identifier(temp_tablename))
    cur.execute(query_drop)
    return None

def pg_copy_from(df, con, tablename, staging_dir, pkey=None,  sep='|'):
    """
    Bulk import into PostgreSql
    - Write the data as a csv file into the staging directory. (without the index)\
    If no primary key is provided:
        - execute  a COPY FROM query
        - delete the file
    Else:
        - Create a temporary empty temp_tablename with same structure as tablename
        - COPY FROM the input data to a temp_tablename
        - INSERT / ON CONFLICT DO NOTHING between temp_tablename and tablename
        - DROP temp_tablename
    Args:
        df (pd.DataFrame): Data to import. All the columns must be in the same order. Index will be copied.
        con (psycopg2.connection): connection object
        staging_dir (str); path of staging_dir
        tablename (str): table name to import
        pkey(str/list): primary key or list. If provided, will allow upsert.
        sep (str): Delimiter

    Returns:
        None
    """
    ###
    filepath = write_csv(df=df, staging_dir=staging_dir, fileprefix=tablename)

    cur = con.cursor()

    if pkey is None:
        copy_wopkey(tablename=tablename, filepath=filepath, cur=cur, sep=sep)

    else:
        copy_withpkey(tablename=tablename, filepath=filepath, cur=cur, sep=sep, pkey=pkey)
    con.commit()
    cur.close()
    os.remove(filepath)
    return None

def pg_insert(df, query, conn):
    """
    Insert data into
    Args:
        df (pd.DataFrame): Data
        query (str): SQL insert query
        conn (psycopg2.connection):

    Returns:
        None
    """
    cur = conn.cursor()
    for i, row in df.iterrows():
        cur.execute(query, list(row))
    conn.commit()
    cur.close()
    return None


def pg_conn(host, dbname, user, password):
    """
    Args:
        host (str):
        dbname (str):
        user (str):
        password (str):
    Returns:
        psycopg2.connection
    """
    conn = psycopg2.connect("host={} dbname={} user={} password={}".format(host, dbname, user, password))
    conn.autocommit = True
    return conn

def pg_engine(host, port, dbname, user, password):
    """
    Args:
        host (str):
        port (str):
        dbname (str):
        user (str):
        password (str):
    Returns:
        psycopg2.connection
    """
    engine = create_engine("postgresql://{}:{}@{}:{}/{}".format(user, password, host, port, dbname))
    return engine
