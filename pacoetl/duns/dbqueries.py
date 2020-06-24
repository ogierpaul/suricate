
q_pg_duns_drop = """DROP TABLE IF EXISTS duns;"""

q_neo_duns_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.duns IS NOT NULL
MERGE(d:Duns{duns:row.duns})
SET
    d.name = row.name,
    d.street = row.street,
    d.postalcode = row.postalcode,
    d.city = row.city,
    d.state = row.state,
    d.countrycode = row.countrycode,
    d.tax1 = row.tax1,
    d.statuscode = row.statuscode,
    d.subscode = row.suscode,
    d.userarea = row.userarea;
"""

q_neo_tax_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.uid IS NOT NULL AND row.duns IS NOT NULL
MERGE(n:Tax{uid:row.uid})
WITH row, n
MATCH (d:Duns{duns:row.duns})
WITH row, n, d
MERGE (d)-[r:HASID]->(n)
SET r.type = row.type;
"""

q_neo_suppliername_load = """
LOAD CSV WITH HEADERS FROM $filename AS row
FIELDTERMINATOR '|'
WITH row
WHERE row.suppliername IS NOT NULL AND row.duns IS NOT NULL
MERGE(n:SupplierName{name:row.suppliername})
WITH row, n
MATCH (d:Duns{duns:row.duns})
WITH row, n, d
MERGE (d)-[r:HASNAME]->(n)
SET r.type = row.type;
"""

