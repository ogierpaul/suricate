import elasticsearch
import time


def _df_to_dump(df, pkey):
    """
    Return each row of the dataframe as a json dump
    Args:
        df (pd.DataFrame):
        pkey (str):

    Returns:
        dict: json dump
    """
    import json
    allrecs = list()
    X = df.copy()
    for i in range(df.shape[0]):
        s = X.iloc[i].dropna().to_dict()
        js = json.dumps(s, default=str)
        if pkey is None:
            allrecs.append({'body': js})
        else:
            allrecs.append({'body': js, 'pkey': s[pkey]})
    return allrecs


def es_index(client, df, index, doc_type, id=None, sleep=1):
    """

    Args:
        client (elasticsearch.Elasticsearch): elastic search client
        df (pd.DataFrame): pd.DataFrame
        index (str): name of es index
        id (str): name of column of index id.
        sleep (int): number of seconds to wait after to complete indexation

    Returns:
        None
    """
    dump = _df_to_dump(df=df, pkey=id)
    if id is None:
        for d in dump:
            client.index(index=index, body=d['body'], doc_type=doc_type)
    else:
        for d in dump:
            client.index(index=index, body=d['body'], id=d['pkey'], doc_type=doc_type)
    time.sleep(sleep)
    return None


def es_create(es_client, indexname, doc_type, mapping):
    """
    Create the index according to the mapping
    Args:
        es_client (elasticsearch.Elasticsearch): elastic search client
        indexname (str): Name of index in ElasticSearch. (also called indice).
        doc_type (str): Name of document type in ElasticSearch. If None, uses the same name as the indexname
        mapping (dict): Mapping

    Returns:
        None
    Examples:
        mapping = {
            "mappings": {
                doc_type: {
                    "properties": {
                        "ariba": {"type": "keyword"},
                        "name": {"type": "text"},
                        "street": {"type": "text"},
                        "city": {"type": "text"},
                        "postalcode": {"type": "keyword"},
                        "state": {"type": "text"},
                        "countrycode": {"type": "keyword"},
                        "iban": {"type": "keyword"},
                        "duns": {"type": "keyword"},
                        "ssn": {"type": "keyword"},
                        "extract_ts": {"type": "date"}
                    }
                }
            }
        }
    """
    es_client.indices.create(index=indexname, doc_type=doc_type, body=mapping, ignore=400)
    return None


def es_create_load(df, es_client, drop, create, indexname, doc_type, pkey, mapping):
    """
    Create the index & load the data into the index
    Args:
        df(pd.DataFrame):
        es_client (elasticsearch.Elasticsearch): elastic search client
        drop (bool): If True, proceed first to drop the index
        create (bool): If True, proceed then to create the index
        indexname (str): Name of index in ElasticSearch. (also called indice).
        doc_type (str): Name of document type in ElasticSearch. If None, uses the same name as the indexname
        mapping (dict): Mapping of the index
        pkey (str): String

    Returns:
        None
    """
    if doc_type is None:
        doc_type = indexname
    if drop is True:
        es_client.indices.delete(index=indexname, ignore=404)
    if create is True:
        es_client.indices.create(index=indexname, body=mapping, doc_type=doc_type, ignore=400)
    es_index(client=es_client, df=df, index=indexname, doc_type=doc_type, id=pkey, sleep=2)
    return None