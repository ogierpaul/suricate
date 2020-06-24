import elasticsearch
import pandas as pd
from suricate.dbconnectors import es_index, es_create
from tutorial.Old.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready
engine = create_engine_ready()


## Load the Right data from sql to put it to ES
# nrows = 200
# df_target = pd.read_sql(sql="SELECT * FROM df_target LIMIT {}".format(nrows), con=engine)
df_target = pd.read_sql(sql="SELECT * FROM df_target", con=engine)
df_target.set_index('ix', drop=True, inplace=True)

## Put the data to ES, drop the index first and then re create
esclient = elasticsearch.Elasticsearch()
es_indice = 'df_target'
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
                    "countrycode": {"type": "keyword"}
                }
            }
        }
    }
    es_create(client=esclient, index='right', mapping=request_body)
    es_index(client=esclient, df=df_target.reset_index(drop=False), index='right', index_id='ix', sleep=5, doc_type="_doc")
pass
catcount = esclient.count(index=es_indice)['count']
assert catcount == df_target.shape[0]
print(catcount)