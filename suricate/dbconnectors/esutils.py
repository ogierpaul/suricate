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
            allrecs.append({'body': js, 'id': s[pkey]})
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
            client.index(index=index, body=d['body'], id=d['id'], doc_type=doc_type)
    time.sleep(sleep)
    return None


def es_create(client, index, mapping):
    """
    Create the index according to the mapping
    Args:
        client (elasticsearch.Elasticsearch): elastic search client
        index (str): Name of index in ElasticSearch. (also called indice).
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
    client.indices.create(index=index, body=mapping, ignore=400)
    return None


def es_create_load(df, client, drop, create, index, doc_type, id, mapping):
    """
    Create the index & load the data into the index
    Args:
        df(pd.DataFrame):
        client (elasticsearch.Elasticsearch): elastic search client
        drop (bool): If True, proceed first to drop the index
        create (bool): If True, proceed then to create the index
        index (str): Name of index in ElasticSearch. (also called indice).
        doc_type (str): Name of document type in ElasticSearch. If None, uses the same name as the indexname
        mapping (dict): Mapping of the index
        id (str): String, name of column to be used as id for the indexation

    Returns:
        None
    """
    if doc_type is None:
        doc_type = index
    if drop is True:
        client.indices.delete(index=index, ignore=404)
    if create is True:
        es_create(client=client, index=index, mapping=mapping)
    es_index(client=client, df=df, index=index, doc_type=doc_type, id=id, sleep=2)
    return None