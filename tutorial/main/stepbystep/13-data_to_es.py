import elasticsearch
import pandas as pd
from suricate.dbconnectors.esconnector import index_with_es
from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
engine = create_engine_ready()

nrows = 200

## Load the Right data from sql
df_right = pd.read_sql(sql="SELECT * FROM df_right LIMIT {}".format(nrows), con=engine)
df_right.set_index('ix', drop=True, inplace=True)

## Put the data to ES, drop the index first and then re create
esclient = elasticsearch.Elasticsearch()
es_indice = 'df_right'
if True:
    try:
        esclient.indices.delete(index=es_indice)
    except:
        pass
    request_body = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 5
        },

        "mappings": {
            "_doc": {
                "properties": {
                    "ix": {"type": "keyword"},
                    "name": {"type": "text"},
                    "street": {"type": "text"},
                    "city": {"type": "text"},
                    "postalcode": {"type": "text"},
                    "countrycode": {"type": "keyword"},
                    "duns": {"type": "text"}
                }
            }
        }
    }
    esclient.indices.create(index=es_indice, body=request_body)
    index_with_es(client=esclient, df=df_right, index=es_indice, ixname="ix", reset_index=True, doc_type='_doc')
    import time
    time.sleep(5)
pass
catcount = esclient.count(index=es_indice)['count']
assert catcount == nrows