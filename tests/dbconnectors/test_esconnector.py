import pytest
from suricate.data.companies import getsource, gettarget
from suricate.dbconnectors.esconnector import EsConnector, unpack_allhits
from suricate.dbconnectors import  es_index, es_create
import elasticsearch
import pandas as pd


@pytest.fixture
def esconnectornew():
    esclient = elasticsearch.Elasticsearch()
    scoreplan = {
        'name': {
            'type': 'FreeText'
        },
        'street': {
            'type': 'FreeText'
        },
        'city': {
            'type': 'FreeText'
        },
        'duns': {
            'type': 'Exact'
        },
        'postalcode': {
            'type': 'FreeText'
        },
        'countrycode': {
            'type': 'Exact'
        }
    }
    escon = EsConnector(
        client=esclient,
        scoreplan=scoreplan,
        index="right",
        explain=False,
        size=10
    )
    return escon

def test_init(esconnectornew):
    assert isinstance(esconnectornew, EsConnector)
    pass

def test_empty_create_index():
    esclient = elasticsearch.Elasticsearch()
    nrows =200
    if True:
        right = gettarget(nrows=nrows)
        try:
            esclient.indices.delete(index='right')
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
        es_create(client=esclient, index='right', mapping=request_body)
        es_index(client=esclient, df=right.reset_index(drop=False), index='right', id='ix', sleep=5, doc_type="_doc")
    pass
    catcount = esclient.count(index="right")['count']
    assert catcount == nrows

def test_index_all():
    esclient = elasticsearch.Elasticsearch()
    right = gettarget(nrows=None)
    es_index(client=esclient, df=right.reset_index(drop=False), index='right', id='ix', doc_type='_doc')
    pass

def test_create_query(esconnectornew):
    df_source = getsource(nrows=10)
    record = df_source.sample().iloc[0]
    q = esconnectornew._write_es_query(record)
    print(q)

def test_getresults(esconnectornew):
    df_source = getsource(nrows=10)
    record = df_source.sample().iloc[0]
    q = esconnectornew._write_es_query(record)
    r = esconnectornew.client.search(body=q, index="right")
    print(r)

def test_search_record(esconnectornew):
    df_source = getsource(nrows=100)
    record = df_source.sample().iloc[0]
    res = esconnectornew.search_record(record=record)
    score = unpack_allhits(res)
    assert len(score) <= esconnectornew.size
    assert isinstance(score, list)
    assert isinstance(score[0], dict)
    print(res)
    print(score)


def test_scorecols_datacols(esconnectornew):
    df_source = getsource(nrows=10)
    for c in df_source.sample(1).index:
        record = df_source.loc[c]
        res = esconnectornew.search_record(record=record)
        score = unpack_allhits(res)
        df = pd.DataFrame.from_dict(score, orient='columns').rename(
                columns={
                    'ix': 'ix_target'
                })
        scorecols = pd.Index(['es_rank', 'es_score'])
        assert df.columns.contains(scorecols[0])
        assert df.columns.contains(scorecols[1])


def test_transform(esconnectornew):
    df_source = getsource(nrows=100)
    X = esconnectornew.fit_transform(X=df_source)
    assert isinstance(X, pd.DataFrame)
    assert X.shape[1] == 2

def test_getsbs(esconnectornew):
    df_source = getsource(nrows=50)
    Xst = esconnectornew.fit_transform(X=df_source)
    ix = Xst.index
    X_sbs = esconnectornew.getsbs(X=df_source, on_ix=ix)
    assert X_sbs.index.equals(ix)
    for c in df_source.columns:
        assert c + '_source' in X_sbs.columns
    return True

