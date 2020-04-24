## DB Connectors
### Es Connector
* EsConnector uses ElasticSearch capabilities to find potential duplicates for a *source* record inside a *target* index

### Data loading and storage
* Source data: DataFrame
* Target Data: Elastic Search Index

### Output
* Similarity matrix:
    * es_score: score of the match in Elastic Search
    * es_rank: rank of the match in Elastic Search
* Side-by-Side view of the source and target data for each potential match

### Usage
See suricate/tests/dbconnectors/test_esconnector.py
