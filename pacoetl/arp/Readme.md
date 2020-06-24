# Arp ETL Module

## Sub-Module description

|Name|Description|
|---|---|
|dbqueries|Database-specific queries for use in etl (SQL queries, ES Mapping, Neo queries..)|
|etl|Functions to move data across repositories|
|transform|Functions to clean and transform the data|

## Transform
- select and rename the columns
- cast the right columns to the right data type
- convert na values to None
- zero padding
- sanitize for JS inputs
- remove confusing values for csv output (remove commas...)
- drop null and duplicates on pkey
- Prefix the ARP number with 'ARP_'
- Clean and Prefix the DNB number with 'DNB_'
- Concat the different names (name, name2) columns in a single name column
- Concat the different street (street, street2, street4) columns in a single street column
- Clean and prefix the tax number codes with the country code
- Re-order the columns

## Load
### Neo4j
- Load the ARP Nodes
- Load the Name nodes (From Name, ARP_harmonized name) and link them to the ARP nodes vie :HASNAME relationship
- Load the Tax nodes (From Eu_vat, Tax1, tax2, Tax3) and link them to the Arp nodes via :HASID relationship
- Load the Cage nodes and link them to the Arp nodes via :HASID relationship
- Load the Duns nodes and link them to the Arp nodes via :HASID relationship


