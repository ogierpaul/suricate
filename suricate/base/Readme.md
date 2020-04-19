# Base Connector Class
* Connector = create a link between two sources of data you wish to deduplicate (source and target)
* Example : DataFrame Connector connects two pandas DataFrames
* Example: Elastic Search Connector connects a source data (pandas Data Frame, source), and a target data (Elastic Search Index, target)

## Purpose
* Create a general framework to deduplicate two sources of data
* The sources of the data could be from different types: pandas DataFrames, elastic search index, postgresql table, or neo4j nodes
* It is useful to have a common structure to be able to use whatever sources of data possible as an input for a machine learning pipeline

## Connector structure:
### Init
* ixname: standardized name for index column, default ix
* source_suffix and target suffix: standardized names to add to identify the columns from the source and target dataframe in a merged table

## Scorer functions: Fit and transform
### Fit
* fit the scorer to the data (example: init tf-idf matrix...)
### Transform
* transform the source and target inputs into a similarity matrix
* to each row corresponds pair of records (one from source data, the other from target data).
* The multiindex from the similarity matrix contains the source and target index values of each of the pairs
* The associated numeric values are the similarity scores

## Visualization
### GetSbs
* Returns a side-by-side view of the data
### Fetch (fetch source, fetch target)
* Returns the data from the source and target data sources according to the index values specified

