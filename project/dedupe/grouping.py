import pandas as pd
import psycopg2
from sqlalchemy import create_engine

from suricate.grouping.new import create_distance_matrix, cluster_from_matrix


def update_neo(df_source, y_proba, y_cluster):
    """

    Returns:

    """
    neo4j_stage = '/Users/paulogier/84-neo4j_home/neo4j-community-3.5.15/import/'
    df_source.to_csv(neo4j_stage + 'df_source.csv', index=True, sep=',', encoding='utf-8')
    z = pd.DataFrame(y_proba).reset_index()
    z['y_discrete'] = pd.cut(z['y_proba'], bins=4, labels=range(4))
    z.to_csv(neo4j_stage + 'y_proba.csv', index=True, sep=',', encoding='utf-8')
    g = y_cluster
    g.name = 'y_cluster'
    pd.DataFrame(y_cluster).to_csv(neo4j_stage + 'y_cluster.csv', index=True, sep=',', encoding='utf-8')
    from py2neo import Graph
    username = 'neo4j'
    password = 'abc123'
    host = 'localhost'
    port = '7687'
    graph = Graph(auth=(username, password), host=host, port=port)
    neo_erase_query = """
    MATCH (n:SupplierGroup)
    DETACH DELETE n;
    """
    neo_erase_similar = """
    MATCH ()-[r:SIMILAR]-()
    DELETE r;
    """
    neo_df_query = """
    LOAD CSV WITH HEADERS FROM "file:///df_source.csv" as row
    MERGE (n:LocalID{uid:row.ix})
    SET
    	n.city = row.city,
        n.name = row.name,
        n.postalCode = row.postalCode,
        n.street = row.street,
        n.country = row.countrycode;
    """
    neo_y_proba_query = """
    LOAD CSV WITH HEADERS FROM "file:///y_proba.csv" as row
    MATCH (s:LocalID{uid:row.ix_source}), (t:LocalID{uid:row.ix_target})
    WHERE row.ix_source <> row.ix_target
    WITH row, s, t
    MERGE (s)-[r:SIMILAR]->(t)
    SET r.score = row.y_proba, r.scorediscrete = row.y_discrete;
    """
    neo_cluster_query = """
    LOAD CSV WITH HEADERS FROM "file:///y_cluster.csv" as row
    MATCH (s:LocalID{uid:row.ix_source})
    WITH s, row
    MERGE (g:SupplierGroup{uid:row.y_cluster})
    WITH s, g
    MERGE (s)-[r:BELONGS]->(g);
    """
    graph.run(neo_erase_query)
    graph.run(neo_erase_similar)
    graph.run(neo_cluster_query)
    graph.run(neo_y_proba_query)
    graph.run(neo_df_query)




if __name__ == '__main__':
    conn = psycopg2.connect("host=127.0.0.1 dbname=suricate user=suricateeditor password=66*$%HWqx*")
    engine = create_engine('postgresql://suricateeditor:66*$%HWqx*@localhost:5432/suricate')
    conn.autocommit = True
    ixnamepairs = ['ix_source', 'ix_target']
    df_source = pd.read_sql('SELECT ix, name, street, city, postalcode, countrycode FROM df_source',
                            con=conn).set_index('ix')
    df_target = pd.read_sql('SELECT ix, name, street, city, postalcode, countrycode  FROM df_target',
                            con=conn).set_index('ix')
    y_proba = pd.read_sql('SELECT ix_source, ix_target, y_proba from results;', con=conn).set_index(ixnamepairs)[
        'y_proba']
    M = create_distance_matrix(y_proba)
    g = cluster_from_matrix(M, threshold=0.80)
    update_neo(df_source, y_proba, g)
