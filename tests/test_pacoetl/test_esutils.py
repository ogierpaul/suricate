import pytest
import pandas as pd
from elasticsearch import Elasticsearch
import datetime
from pacoetl.utils.esutils import es_index



@pytest.fixture
def index_name():
    return 'test-index'

@pytest.fixture
def doc_type():
    return 'supplier'

@pytest.fixture
def data_raw():
    df = pd.DataFrame([
        [1, 'What makes you think she is a witch', 1.0, '123456'],
        [2, "A scratch? Your arm's off!", 2.0, '654321'],
        ['c', 'what have the Romans ever done for us', None, '214365'],
        [4, 'Romans go home!', 3.5, None]
    ],
        columns=['pkey', 'name', 'value', 'keyword'])
    df['ts'] = datetime.datetime.now()
    df['ts'] = df['ts'].apply(lambda r: r.isoformat())
    return df


@pytest.fixture
def es_client():
    return Elasticsearch()

def reset_index(e, index_name, doc_type):
    e.indices.delete(index=index_name, ignore=404)
    mymapping = {
        "mappings": {
            doc_type: {
                "properties": {
                    "pkey": {"type": "keyword"},
                    "name": {"type": "text"},
                    "value": {"type": "double"},
                    "ts": {"type": "date"}
                }
            }
        }
    }
    e.indices.create(index=index_name, body=mymapping, ignore=400)
    return None


def test_create_index(es_client, index_name, doc_type):
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
    e = es_client
    reset_index(e, index_name=index_name, doc_type=doc_type)
    assert e.indices.exists(index=index_name)


def test_insert_document_wo_id(es_client, data_raw, index_name, doc_type):
    reset_index(es_client, index_name=index_name, doc_type=doc_type)
    es_index(client=es_client, df=data_raw, index=index_name, doc_type=doc_type)
    assert es_client.count(index='test-index')['count'] == 4

def test_insert_document_with_id(es_client, data_raw, index_name, doc_type):
    reset_index(es_client, index_name=index_name, doc_type=doc_type)
    es_index(client=es_client, df=data_raw, index=index_name, doc_type=doc_type, id='pkey')
    assert es_client.count(index='test-index')['count'] == 4

def test_upsert_document_with_id(es_client, data_raw, index_name, doc_type):
    reset_index(es_client, index_name=index_name, doc_type=doc_type)
    df = data_raw.copy()
    df.loc[df['pkey'] == 4, 'pkey'] = 1
    es_index(client=es_client, df=df, index=index_name, doc_type=doc_type, id='pkey')
    assert es_client.count(index='test-index')['count'] == 3


