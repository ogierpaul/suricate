import datetime

import pacoetl.arp
from pacoetl.utils import pg_engine, pg_conn, neo_graph, create_batch, es_client, pg_copy_from, staging_dir
from pacoetl.utils.others import uniquepairs, printmessage
import pandas as pd

from pacoetl.arp.etl import global_indexname
from suricate.dbconnectors import EsConnector
import numpy as np
global_ixnamepairs = ['arp_source', 'arp_target']


def create():
    con = pg_conn()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS arp_ixp (
    ixp VARCHAR,
    arp_source VARCHAR,
    arp_target VARCHAR,
    es_score DOUBLE PRECISION,
    PRIMARY KEY (ixp)
)
    """)
    con.commit()
    cur.close()
    con.close()
    return True


def es_neighbours():
    df = pd.read_sql(pacoetl.arp.dbqueries.q_pg_arp_select, con=pg_engine()).set_index(['arp'])
    print(datetime.datetime.now(), ' | data loaded')
    dfb = create_batch(df, batch_size=2000)
    index_name = global_indexname
    doc_type = index_name
    scoreplan = {
        'arp': {'type': 'FreeText'},
        'name': {'type': 'FreeText'},
        'street': {'type': 'FreeText'},
        'city': {'type': 'FreeText'},
        'postalcode': {'type': 'FreeText'},
        'state': {'type': 'Exact'},
        'countrycode': {'type': 'Exact'},
        'duns': {'type': 'Exact'},
        'eu_vat': {'type': 'FreeText'},
        'tax1': {'type': 'FreeText'},
        'tax2': {'type': 'FreeText'},
        'tax3': {'type': 'FreeText'},
        'cage': {'type': 'Exact'},
        'arp_harmonizedname': {'type': 'Exact'},
        'arp_partnercompany': {'type': 'Exact'},
        'concatenatedids': {'type': 'FreeText'}
    }
    escon = EsConnector(
        client=es_client(),
        scoreplan=scoreplan,
        index=index_name,
        explain=False,
        size=10,
        doc_type=doc_type,
        ixname=index_name
    )
    cols = ['arp_source', 'arp_target', 'es_score']
    for i, d in enumerate(dfb):
        print(datetime.datetime.now(), ' | Start batch {} of {}'.format(i + 1, len(dfb)))
        Xsm = escon.fit_transform(X=d).reset_index(drop=False)[cols]
        print(datetime.datetime.now(), ' | ES scores calculated')
        Xsm.to_sql('arp_es_neighbours', if_exists='append', con=pg_engine())
        print(datetime.datetime.now(), ' | loaded in PG\n')


def neo_neighbours():
    def updaters(s, r):
        if len(r) > 0:
            for i in r:
                s.add(i['m'])
        return s

    def updatedf(df):
        q_neo_name1 = """
        MATCH (s:Arp{arp:$arp_source})-[:HASNAME]-(t:Arp)
        WHERE t.arp <> s.arp
        RETURN DISTINCT(t.arp) AS m
        """

        q_neo_name2 = """
        MATCH (s:Arp{arp:$arp_source})-[:HASNAME]-(t:Arp)-[:HASID]-(t2:Arp)
        WHERE t2.arp <> s.arp
        RETURN DISTINCT(t2.arp) AS m
        """

        q_neo_id1 = """
        MATCH (s:Arp{arp:$arp_source})-[:HASID*0..2]-(t:Arp)
        WHERE s.arp <> t.arp
        RETURN DISTINCT(t.arp) AS m
        """
        cols = ['arp_source', 'arp_target']
        data = pd.DataFrame(columns=cols)
        for a in df['arp_source']:
            matches = set()
            matches = updaters(matches, graph.run(q_neo_name1, parameters={'arp_source': a}).data())
            matches = updaters(matches, graph.run(q_neo_name2, parameters={'arp_source': a}).data())
            matches = updaters(matches, graph.run(q_neo_id1, parameters={'arp_source': a}).data())
            d = pd.Series(data=list(matches), name='arp_target')
            d = pd.DataFrame(d)
            d['arp_source'] = a
            data = pd.concat([data, d[cols]], axis=0, ignore_index=True)
        return data

    graph = neo_graph()
    df = pd.read_sql("SELECT arp AS arp_source FROM arp;", con=pg_engine())
    dfb = create_batch(df)
    for i, d in enumerate(dfb):
        print(datetime.datetime.now(), ' | Start batch {} of {}'.format(i + 1, len(dfb)))
        data = updatedf(d)
        print(datetime.datetime.now(), ' | Neo pairs calculated')
        data.to_sql('arp_neo_neighbours', if_exists='append', con=pg_engine(), index=False)
        print(datetime.datetime.now(), ' | loaded in PG\n')
    return None


def update_ixp():
    df1 = pd.read_sql('SELECT arp_source, arp_target, es_score FROM arp_es_neighbours;', con=pg_conn())
    df2 = pd.read_sql('SELECT arp_source, arp_target, NULL AS es_score FROM arp_neo_neighbours;', con=pg_conn())
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    ixnamepairs = ['arp_source', 'arp_target']
    colname = 'es_score'
    df = df.loc[df[ixnamepairs[0]] != df[ixnamepairs[1]]]
    df['ixp'] = df.apply(lambda r: uniquepairs(r[ixnamepairs[0]], r[ixnamepairs[1]]), axis=1)
    print(df.head())
    y = df.pivot_table(index='ixp', values=colname, aggfunc=np.nanmean)
    y.columns = [colname]
    df = df.drop_duplicates(subset=['ixp']).drop([colname], axis=1).join(y, on='ixp', how='left')
    df = df[['ixp', 'arp_source', 'arp_target', 'es_score']]
    df.set_index('ixp', inplace=True)
    print(df.head())
    pg_copy_from(df=df, tablename='arp_ixp', staging_dir=staging_dir, con=pg_conn(), pkey='ixp', sep='|')


def get_single_score(df, col, index_name, doc_type, ixnamepairs):
    ixp = df.index
    uniquesources = df.drop_duplicates(subset=[ixnamepairs[0]]).dropna(subset=[col]).set_index(ixnamepairs[0])[[col]]
    unique_names = uniquesources.drop_duplicates(subset=[col]).set_index([col], drop=False)
    uniquesources.rename(columns={col: col+'_source'}, inplace=True)
    unique_names.index.name = 'arp'
    printmessage('unique values {} from df:{}'.format(unique_names.shape[0], df.shape[0]))
    dfb = create_batch(unique_names, batch_size=1000)
    scoreplan = {
        col: {'type': 'FreeText'}
    }
    escon = EsConnector(
        client=es_client(),
        scoreplan=scoreplan,
        index=index_name,
        explain=False,
        size=20,
        doc_type=doc_type,
        ixname=index_name
    )
    cols = ixnamepairs + ['es_score']
    alldata = None
    for i, d in enumerate(dfb):
        print(datetime.datetime.now(), ' | Start batch {} of {}'.format(i + 1, len(dfb)))
        Xsm = escon.fit_transform(X=d).reset_index(drop=False)[cols]
        Xsm.rename(columns={'es_score': col + '_es', index_name+'_source':col+'_source'}, inplace=True)
        print(datetime.datetime.now(), ' | ES scores calculated with {} rows'.format(Xsm.shape[0]))
        if alldata is None:
            alldata = Xsm.copy()
            del Xsm
        else:
            alldata = pd.concat([alldata, Xsm], ignore_index=True, axis=0)
            del Xsm
    alldata.set_index(col+'_source', inplace=True)
    df2 = uniquesources.join(alldata, on=[col+'_source'], how='left')
    del alldata
    df2.reset_index(drop=False, inplace=True)
    df2 = df2[ixnamepairs+[col+'_es']]
    df2['ixp'] = df2[ixnamepairs[0]] + '-' + df2[ixnamepairs[1]]
    df2 = df2.set_index('ixp')
    ix_common = df2.index.intersection(ixp)
    df2 = df2.loc[ix_common]
    return df2


def es_name_street():
    q = """
    SELECT a.ixp, arp_source, name, street FROM
    (SELECT ixp, arp_source, arp_target FROM arpdedupe.arp_ixp) a
    LEFT JOIN (SELECT arp as arp_source, name, street FROM paco.arp) b USING (arp_source)
    """
    df = pd.read_sql(q, con=pg_engine()).set_index(['ixp'])[['arp_source']]
    print(datetime.datetime.now(), ' | data loaded')
    index_name = global_indexname
    doc_type = index_name
    name_scores = get_single_score(df=df, index_name=index_name, doc_type=doc_type, col='name', ixnamepairs=global_ixnamepairs)

    street_scores = get_single_score(df=df, index_name=index_name, doc_type=doc_type, col='street', ixnamepairs=global_ixnamepairs)
    df = df.join(name_scores, how='left')
    df = df.join(street_scores, how='left')
    return df

if __name__ == '__main__':
    q = """
    SELECT a.ixp, arp_source, name, street FROM
    (SELECT ixp, arp_source FROM arpdedupe.arp_ixp  WHERE es_score>20) a
    LEFT JOIN (SELECT arp as arp_source, name, street FROM paco.arp) b USING (arp_source)
    """
    df = pd.read_sql(q, con=pg_engine()).set_index(['ixp'])
    print(datetime.datetime.now(), ' | data loaded')
    index_name = global_indexname
    doc_type = index_name
    # name_scores = get_single_score(df=df, index_name=index_name, doc_type=doc_type, col='name', ixnamepairs=global_ixnamepairs)
    #
    # street_scores = get_single_score(df=df, index_name=index_name, doc_type=doc_type, col='street', ixnamepairs=global_ixnamepairs)
    # df = df.join(name_scores, how='left')
    # df = df.join(street_scores, how='left')
    col = 'name'
    ixnamepairs = global_ixnamepairs
    df_name = get_single_score(df, 'name', global_indexname, global_indexname, ixnamepairs)
    df_street = get_single_score(df, 'street', global_indexname, global_indexname, ixnamepairs)
    df_es = pd.DataFrame(index=df.index).join(df_name[['name_es']], how='left').join(df_street[['street_es']], how='left')
    df_es.to_sql('arp_namestreetes', schema='arpdedupe', con=pg_engine(), index=True)





