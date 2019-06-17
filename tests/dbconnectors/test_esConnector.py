import pytest
from suricate.data.dataframes import df_left
from suricate.dbconnectors.esconnector import EsConnector, unpack_allhits
import elasticsearch
import pandas as pd


@pytest.fixture
def esconnector():
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
        client_es=esclient,
        scoreplan=scoreplan,
        index_es="pg-suppliers-all",
        doc_type_es='supplierid',
        explain=True,
        size=20
    )
    return escon

def test_init(esconnector):
    assert isinstance(esconnector, EsConnector)
    pass

def test_search_record(df_left, esconnector):
    record = df_left.sample().iloc[0]
    res = esconnector.search_record(record=record)
    score = unpack_allhits(res)
    assert len(score) == esconnector.size
    assert isinstance(score, list)
    assert isinstance(score[0], dict)

def test_searchtodf(df_left, esconnector):
    for c in df_left.sample(50).index:
        record = df_left.loc[c]
        res = esconnector.search_record(record=record)
        score = unpack_allhits(res)
        df = pd.DataFrame(score)
        assert True

def test_scorecols_datacols(df_left, esconnector):
    for c in df_left.sample(1).index:
        record = df_left.loc[c]
        res = esconnector.search_record(record=record)
        score = unpack_allhits(res)
        df = pd.DataFrame(score)
        usecols = df_left.columns.intersection(df.columns).union(pd.Index([df_left.index.name]))
        scorecols = pd.Index(['es_rank', 'es_score'])
        print(scorecols)
        assert True


