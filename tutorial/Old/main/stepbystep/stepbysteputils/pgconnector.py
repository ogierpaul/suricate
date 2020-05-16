import json
from sqlalchemy import create_engine
import pandas as pd
with open('/tutorial/Old/main/stepbystep/stepbysteputils/config.json') as dbconfig:
    jdb = json.load(dbconfig)

def create_engine_ready():
    engine = create_engine(
        'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
            jdb['postgres']['username'],
            jdb['postgres']['password'],
            jdb['postgres']['host'],
            jdb['postgres']['port'],
            jdb['postgres']['database'],
        )
    )
    return engine

def work_on_ix(ix, engine):
    df = pd.DataFrame(index=ix)
    df.index.name = 'ix'
    df.to_sql(name='tempix', con=engine, if_exists='replace', index=True)
    return True


def select_from_ix(table, engine):
    query = f"""
    SELECT *
    FROM
      {table}
    INNER JOIN tempix USING(ix)
    """
    df = pd.read_sql(query, con=engine)
    return df

def select_from_ix(table, ix):
    work_on_ix(ix)
    return select_from_ix(table=table)