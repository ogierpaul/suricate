import datetime
import os

import psycopg2
from psycopg2 import sql

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

def write_csv(df, staging_dir, filename, tablename):
    """

    Args:
        df (pd.DataFrame)
        staging_dir (str):
        filename (str):
        tablename (str):

    Returns:
        str: filepath of the csv written
    """
    if filename is None:
        filename = tablename + '_' + datetime.datetime.now().strftime("%Y-%b-%d-%H-%M-%S") + '.csv'
    filepath = staging_dir + '/' + filename
    df.to_csv(path_or_buf=filepath, encoding='utf-8', sep='|', index=False)
    return filepath

def copy_wopkey(tablename, filepath, cur, sep='|'):
    """
    copy from filepath to tablename with delimiter sep
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

def pg_copy_from(df, conn, tablename, staging_dir, pkey=None, filename=None, sep='|'):
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
        df (pd.DataFrame): Data to import. All the columns must be in the same order. Index will not be copied.
        conn (psycopg2.connection): connection object
        staging_dir (str); path of staging_dir
        tablename (str): table name to import
        filename (str): name of the file. If none, will use timestamp of the time when the function is called
        pkey(str/list): primary key or list. If provided, will allow upsert.
        sep (str): Delimiter

    Returns:
        None
    """
    ###
    filepath = write_csv(df=df, staging_dir=staging_dir, tablename=tablename, filename=filename)

    cur = conn.cursor()

    if pkey is None:
        copy_wopkey(tablename=tablename, filepath=filepath, cur=cur, sep=sep)

    else:
        copy_withpkey(tablename=tablename, filepath=filepath, cur=cur, sep=sep, pkey=pkey)
    conn.commit()
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


def pg_conn():
    """
    Returns:
        psycopg2.connection
    """
    conn = psycopg2.connect("host=127.0.0.1 dbname=pacoetl user=cedar password=abc123")
    conn.autocommit = True
    return conn