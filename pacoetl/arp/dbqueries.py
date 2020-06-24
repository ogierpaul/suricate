from psycopg2 import sql

q_pg_arp_drop = sql.SQL(""" DROP TABLE IF EXISTS arp;""")

q_pg_arp_select = """
SELECT
    arp,
    name,
    street,
    postalcode,
    city,
    state,
    countrycode,
    duns,
    eu_vat,
    tax1,
    tax2,
    tax3,
    arp_harmonizedname,
    arp_partnercompany,
    cage,
    concatenatedids
FROM
    arp;
"""

q_pg_arphid_drop = sql.SQL("""
DROP TABLE IF EXISTS arp_hid;
""")

q_neo_arp_drop = """
MATCH (n:Arp)
DETACH DELETE n;
"""

q_neo_arp_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.arp IS NOT NULL
MERGE(a:Arp{arp:row.arp})
ON CREATE SET
      a.name = row.name,
      a.street = row.street,
      a.city = row.city,
      a.postalcode = row.postalcode,
      a.countrycode = row.countrycode
ON MATCH SET
      a.name = row.name,
      a.street = row.street,
      a.city = row.city,
      a.postalcode = row.postalcode,
      a.countrycode = row.countrycode;
"""


q_pg_arpytrue_drop = sql.SQL(
    """
    DROP TABLE IF EXISTS  arp_ytrue;
    """

)

q_pg_arpytrue_create = sql.SQL("""
CREATE TABLE IF NOT EXISTS arp_ytrue (
    arp_source VARCHAR,
    arp_target VARCHAR,
    y_true INTEGER,
    PRIMARY KEY (arp_source, arp_target)
);
""")

q_pg_arpytrue_select = """
SELECT arp_source, arp_target, y_true FROM arp_ytrue
"""

q_pg_arpytrue_update = sql.SQL("""
INSERT INTO arp_ytrue (arp_source, arp_target, y_true) 
VALUES (%s, %s, %s) 
ON CONFLICT (arp_source, arp_target)
DO UPDATE
SET y_true = excluded.y_true ;
""")

q_pg_arppossiblepairs_convert = """
SELECT arp_source,
       arp_target,
       (CASE WHEN y_proba >0.5 THEN 1 ELSE 0 END) as y_true
FROM arp_possiblepairs
"""

q_neo_cage_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.cage IS NOT NULL AND row.arp IS NOT NULL
MERGE(n:Cage{cage:row.cage})
WITH row, n
MATCH (a:Arp{arp:row.arp})
WITH row, n, a
MERGE (a)-[:HASID]->(n);
"""

q_neo_cage_drop = """
MATCH (n:Cage)
DETACH DELETE n;
"""

q_neo_duns_drop = """
MATCH (n:Duns)
DETACH DELETE n;
"""
q_neo_duns_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.duns IS NOT NULL  AND row.arp IS NOT NULL
MERGE(n:Duns{duns:row.duns})
WITH row, n
MATCH (a:Arp{arp:row.arp})
WITH row, n, a
MERGE (a)-[r:HASID]->(n)
SET r.origin = 'ARP';
"""

q_neo_tax_drop = """
MATCH (n:Tax)
DETACH DELETE n;
"""
q_neo_tax_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.uid IS NOT NULL AND row.arp IS NOT NULL
MERGE(n:Tax{uid:row.uid})
WITH row, n
MATCH (a:Arp{arp:row.arp})
WITH row, n, a
MERGE (a)-[r:HASID]->(n)
SET r.type = row.type;
"""

q_neo_suppliername_drop = """
MATCH (n:SupplierName)
DETACH DELETE n;
"""
q_neo_suppliername_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.suppliername IS NOT NULL AND row.arp IS NOT NULL
MERGE(n:SupplierName{name:row.suppliername})
WITH row, n
MATCH (a:Arp{arp:row.arp})
WITH row, n, a
MERGE (a)-[r:HASNAME]->(n)
SET r.type = row.type;
"""

q_neo_arppartnercompany_drop = """
MATCH (n:ArpPartnerCompany)
DETACH DELETE n;
"""

q_neo_arppartnercompany_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.arp_partnercompany IS NOT NULL AND row.arp IS NOT NULL
MERGE(n:ArpPartnerCompany{arp_partnercompany :row.arp_partnercompany})
WITH row, n
MATCH (a:Arp{arp:row.arp})
WITH row, n, a
MERGE (a)-[:HASID]->(n);
"""

q_neo_supplierhid_drop = """
MATCH (n:SupplierHarmonized)
DETACH DELETE n;
"""
q_neo_supplierhid_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.hid IS NOT NULL AND row.arp IS NOT NULL
MERGE(n:SupplierHarmonized{hid :row.hid})
WITH row, n
MATCH (a:Arp{arp:row.arp})
WITH row, n, a
MERGE (a)-[:BELONGS]->(n);
"""

q_neo_possiblepairs_delete = """
MATCH (s:Arp)-[r:SIMILAR]-(t:Arp)
DELETE r;
"""

q_neo_possiblepairs_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.arp_source <> row.arp_target
MATCH (s:Arp{arp:row.arp_source}), (t:Arp{arp:row.arp_target})
WITH row, s, t
MERGE (s)-[r:SIMILAR]->(t)
SET r.score = row.y_proba, r.scorediscrete = row.y_discrete;
"""

q_neo_arpytrue_delete = """
MATCH (s:Arp)-[r:SAME]-(t:Arp)
DELETE r;
"""

q_neo_arpytrue_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.arp_source <> row.arp_target
MATCH (s:Arp{arp:row.arp_source}), (t:Arp{arp:row.arp_target})
WITH row, s, t
MERGE (s)-[r:SAME]->(t)
"""
