import pandas as pd
from suricate.dbconnectors import EsConnector
from pacoetl.utils import pg_engine, pg_conn, es_client, neo_graph, create_batch, pg_copy_from
from pacoetl.utils.others import uniquepairs, printmessage
import datetime
import numpy as np

global_target_name = 'arp'

def find_neighbours_es():
    q = """
    SELECT
        ariba,
        name,
        street,
        postalcode,
        city,
        state,
        countrycode,
        eu_vat,
        tax1,
        arp,
        duns,
        cage
    FROM eproc;
    """
    df = pd.read_sql(q, con=pg_engine()).set_index('ariba')
    df['concatenatedsids'] = df.apply(
        lambda r:';'.join(r[['eu_vat', 'tax1', 'tax_siret', 'tax_nif', 'tax_steuernummer']].dropna()), axis=1)


    esc = EsConnector(
        client=es_client(),
        index=global_target_name,
        scoreplan={
            'arp': {'type': 'Exact'},
            'name': {'type': 'FreeText'},
            'street': {'type': 'FreeText'},
            'city': {'type': 'FreeText'},
            'postalcode': {'type': 'FreeText'},
            'state': {'type': 'Exact'},
            'countrycode': {'type': 'Exact'},
            'duns': {'type': 'Exact'},
            'eu_vat': {'type': 'keyword'},
            'tax1': {'type': 'FreeText'},
            'concatenatedids': {'type': 'FreeText'}
        },
        doc_type=global_target_name,
        size=10,
        explain=True,
        ixname='arp'
    )
    df.index.name = 'arp'
    Xsm = esc.fit_transform(df)
    Xsm = Xsm.reset_index(drop=False).rename(columns={'arp_source':'ariba_source'})
    Xsm['ixp']= Xsm[['ariba_source', 'arp_target']].apply(lambda r:'-'.join(r), axis=1)
    Xsm = Xsm [['ixp','ariba_source', 'arp_target', 'es_score']]
    Xsm.to_sql('eprocarpneighbours_es', index=False, con=pg_engine(), schema='eprocarp')
    return True

def findneo():
    def updaters(s, r):
            if len(r) > 0:
                for i in r:
                    s.add(i['m'])
            return s

    def updatedf(df):
        q_neo_name1 = """
            MATCH (s:eProc{ariba:$ariba_source})-[:HASNAME]-(t:Arp)
            RETURN DISTINCT(t.arp) AS m
            """

        q_neo_name2 = """
            MATCH (s:eProc{ariba:$ariba_source})-[:HASNAME]-(t:Arp)-[:HASID]-(t2:Arp)
            RETURN DISTINCT(t2.arp) AS m
            """
        q_neo_id1 = """
            MATCH (s:eProc{ariba:$ariba_source})-[:HASID*0..2]-(t:Arp)
            RETURN DISTINCT(t.arp) AS m
            """

        cols = ['ariba_source', 'arp_target']
        data = pd.DataFrame(columns=cols)

        for a in df['ariba_source']:
            matches = set()
            matches = updaters(matches, graph.run(q_neo_name1, parameters={'ariba_source': a}).data())
            matches = updaters(matches, graph.run(q_neo_name2, parameters={'ariba_source': a}).data())
            matches = updaters(matches, graph.run(q_neo_id1, parameters={'ariba_source': a}).data())
            d = pd.Series(data=list(matches), name='arp_target')
            d = pd.DataFrame(d)
            d['ariba_source'] = a
            data = pd.concat([data, d[cols]], axis=0, ignore_index=True)
        return data

    graph = neo_graph()
    df = pd.read_sql("SELECT ariba AS ariba_source FROM eproc;", con=pg_engine())
    dfb = create_batch(df)
    for i, d in enumerate(dfb[:1]):
        print(datetime.datetime.now(), ' | Start batch {} of {}'.format(i + 1, len(dfb)))
        data = updatedf(d)
        print(datetime.datetime.now(), ' | Neo pairs calculated')
        data['ixp'] = data[['ariba_source', 'arp_target']].apply(lambda r: '-'.join(r), axis=1)
        data = data[['ixp', 'ariba_source', 'arp_target']]
        # print(datetime.datetime.now(), ' | loaded in PG\n')
    data.to_sql('eprocarpneighbours_neo', if_exists='append', con=pg_engine(), schema='eprocarp',index=False)


def create_unique_ixp():
    df1 = pd.read_sql('SELECT ariba_source, arp_target, es_score FROM eprocarp.eprocarpneighbours_es;', con=pg_conn())
    df2 = pd.read_sql('SELECT ariba_source, arp_target, NULL as es_score FROM eprocarp.eprocarpneighbours_neo;', con=pg_conn())
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    ixnamepairs = ['ariba_source', 'arp_target']
    colname = 'es_score'
    df = df.loc[df[ixnamepairs[0]] != df[ixnamepairs[1]]]
    df['ixp'] = df.apply(lambda r: uniquepairs(r[ixnamepairs[0]], r[ixnamepairs[1]]), axis=1)
    print(df.head())
    y = df.pivot_table(index='ixp', values=colname, aggfunc=np.nanmean)
    y.columns = [colname]
    df = df.drop_duplicates(subset=['ixp']).drop([colname], axis=1).join(y, on='ixp', how='left')
    df = df[['ixp', 'ariba_source', 'arp_target', 'es_score']]
    df.set_index('ixp', inplace=True)
    df.order_values(by='ixp', ascending=False, inplace=True)
    print(df.head())
    df.to_sql('eprocarp_ixp', con=pg_engine(), schema='eprocarp')
    return None

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

if __name__ == '__main__':
    q = """
    SELECT a.ixp, arp_source, name, street FROM
    (SELECT ixp, ariba_source AS arp_source FROM eprocarp.eprocarp_ixp  WHERE es_score>20) a
    LEFT JOIN (SELECT ariba as arp_source, name, street FROM paco.eproc) b USING (arp_source)
    """
    df = pd.read_sql(q, con=pg_engine()).set_index(['ixp'])
    print(datetime.datetime.now(), ' | data loaded')
    index_name = 'arp'
    doc_type = index_name
    # name_scores = get_single_score(df=df, index_name=index_name, doc_type=doc_type, col='name', ixnamepairs=global_ixnamepairs)
    #
    # street_scores = get_single_score(df=df, index_name=index_name, doc_type=doc_type, col='street', ixnamepairs=global_ixnamepairs)
    # df = df.join(name_scores, how='left')
    # df = df.join(street_scores, how='left')
    col = 'name'
    ixnamepairs = ['arp_source', 'arp_target']
    df_name = get_single_score(df, 'name', index_name, doc_type, ixnamepairs)
    df_street = get_single_score(df, 'street', index_name, doc_type, ixnamepairs)
    df_es = pd.DataFrame(index=df.index).join(df_name[['name_es']], how='left').join(df_street[['street_es']], how='left')
    df_es.to_sql('eprocarp_namestreet_es', schema='eprocarp', con=pg_engine(), index=True)

