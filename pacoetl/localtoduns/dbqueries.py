q_pg_localidduns_drop = """
DROP TABLE IF EXISTS  localsuppliers_duns;
"""

q_neo_arpduns_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.sysid IS NOT NULL AND row.localid IS NOT NULL AND row.duns IS NOT NULL
MERGE (d:Duns{duns:row.duns})
WITH row, d
MATCH (a:Arp{arp:row.localid})
WITH row, d, a
MERGE (a)-[r:HASID]->(d)
SET r.duns = 'True';
"""
