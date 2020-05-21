import pandas as pd
import pytest
import psycopg2
import datetime
from pacoetl.utils import pg_insert, pg_copy_from

@pytest.fixture
def pg_conn():
    """
    Connect to the Sparkifydb using username and password provided by Udacity
    Returns:
        psycopg2.connection
    """
    conn = psycopg2.connect("host=127.0.0.1 dbname=paco user=cedar password=abc123")
    conn.autocommit = True
    return conn

@pytest.fixture
def raw_data():
    df = pd.DataFrame([
        [1, 'foo' ,2.5],
        [2, 'bar',],
        [3, 'baz', 1.0],
        [1, 'duplicate_foo', 2.0],
    ],
        columns=['id', 'name', 'value']
    )
    df['ts'] = datetime.datetime.now()
    return df

def _create_table(conn):
    create_sql = """
    CREATE TABLE IF NOT EXISTS test_foo (
        id INTEGER,
        name VARCHAR(16),
        value DOUBLE PRECISION,
        ts TIMESTAMP,
        PRIMARY KEY (id)
    );
    """
    delete_sql = """
    DELETE FROM test_foo;
    """
    cur = conn.cursor()
    cur.execute(create_sql)
    cur.execute(delete_sql)
    conn.commit()
    cur.close()
    return None

def _drop_table(conn):
    cur = conn.cursor()
    cur.execute("""DROP TABLE test_foo;""")
    conn.commit()
    cur.close()
    return None

def _select_test(conn, expected):
    cur = conn.cursor()
    select_sql = """
    SELECT COUNT(*) FROM test_foo;
    """
    cur.execute(select_sql)
    r = cur.fetchone()
    conn.commit()
    cur.close()
    test_result = (r[0] == expected)
    return test_result

def test_insert(raw_data, pg_conn):
    _create_table(conn=pg_conn)
    upsert_sql = """
    INSERT INTO test_foo(id, name, value, ts) 
    VALUES (%s, %s, %s, %s) 
    ON CONFLICT (id)
    DO UPDATE
    SET value = excluded.value ;
    """

    df = raw_data
    cur = pg_conn.cursor()
    pg_insert(df=df, query=upsert_sql, conn=pg_conn)
    pg_conn.commit()
    assert _select_test(conn=pg_conn, expected=3)
    cur.execute("""DROP TABLE test_foo;""")
    pg_conn.commit()
    cur.close()
    pg_conn.close()
    assert True

def test_copy_from(raw_data, pg_conn):
    _create_table(conn=pg_conn)
    df = raw_data
    cur = pg_conn.cursor()
    staging_dir = '../../project/data_dir/staging'
    pg_copy_from(df=df, conn=pg_conn, tablename='test_foo', staging_dir=staging_dir, pkey='id')
    pg_conn.commit()
    assert _select_test(conn=pg_conn, expected=3)
    cur.execute("""DROP TABLE test_foo;""")
    pg_conn.commit()
    cur.close()
    pg_conn.close()
    assert True








