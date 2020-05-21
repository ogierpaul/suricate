from psycopg2 import sql

pg_create_goldenrecord = sql.SQL("""
CREATE TABLE IF NOT EXISTS goldenrecord (
    gid VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP
),
PRIMARY KEY (gid);
""")
pg_delete_goldenrecord = sql.SQL(""" DROP TABLE IF EXISTS goldenrecord;""")


create_arp = sql.SQL("""
CREATE TABLE IF NOT EXISTS arp (
    arp VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    duns VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP
),
PRIMARY KEY (arp);
""")
delete_arp = sql.SQL(""" DROP TABLE IF EXISTS arp;""")

create_eproc = sql.SQL("""
CREATE TABLE IF NOT EXISTS eproc (
    ariba VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    arp VARCHAR,
    duns VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP
),
PRIMARY KEY (ariba);
""")
delete_eproc = sql.SQL(""" DROP TABLE IF EXISTS eproc;""")

create_duns = sql.SQL("""
CREATE TABLE IF NOT EXISTS duns (
    duns VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP
),
PRIMARY KEY (duns);
""")
delete_duns = sql.SQL(""" DROP TABLE IF EXISTS duns;""")

create_localsuppliers = sql.SQL("""
CREATE TABLE IF NOT EXISTS localsuppliers (
    uid VARCHAR,
    sysid VARCHAR,
    localid VARCHAR,
    gid VARCHAR,
    name VARCHAR,
    street VARCHAR,
    postalcode VARCHAR,
    city VARCHAR,
    state VARCHAR,
    countrycode VARCHAR,
    arp VARCHAR,
    duns VARCHAR,
    eu_vat VARCHAR,
    tax1 VARCHAR,
    tax2 VARCHAR,
    tax3 VARCHAR,
    created_ts TIMESTAMP,
    extract_ts TIMESTAMP,
    updated_ts TIMESTAMP
),
PRIMARY KEY (uid);
""")
delete_localsuppliers = sql.SQL(""" DROP TABLE IF EXISTS localsuppliers;""")



