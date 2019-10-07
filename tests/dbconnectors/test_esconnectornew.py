import pytest
from suricate.data.companies import getleft
from suricate.dbconnectors.esconnectornew import EsConnectorNew, unpack_allhits
import elasticsearch
import pandas as pd
import numpy as np

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
    escon = EsConnectorNew(
        client=esclient,
        scoreplan=scoreplan,
        index="right",
        explain=False,
        size=10
    )
    return escon

def test_init(esconnectornew):
    assert isinstance(esconnectornew, EsConnectorNew)
    pass

def test_create_query(esconnectornew):
    df_left = getleft(nrows=10)
    record = df_left.sample().iloc[0]
    q = esconnectornew._write_es_query(record)
    print(q)

def test_getresults(esconnectornew):
    df_left = getleft(nrows=10)
    record = df_left.sample().iloc[0]
    q = esconnectornew._write_es_query(record)
    r = esconnectornew.client.search(body=q, index="right")
    print(r)

def test_search_record(esconnectornew):
    df_left = getleft(nrows=100)
    record = df_left.sample().iloc[0]
    res = esconnectornew.search_record(record=record)
    score = unpack_allhits(res)
    assert len(score) <= esconnectornew.size
    assert isinstance(score, list)
    assert isinstance(score[0], dict)
    print(res)
    print(score)


def test_scorecols_datacols(esconnectornew):
    df_left = getleft(nrows=100)
    for c in df_left.sample(1).index:
        record = df_left.loc[c]
        res = esconnectornew.search_record(record=record)
        score = unpack_allhits(res)
        df = pd.DataFrame(score)
        usecols = df_left.columns.intersection(df.columns).union(pd.Index([df_left.index.name]))
        scorecols = pd.Index(['es_rank', 'es_score'])
        print(scorecols)
        assert True

def test_transform(esconnector):
    df_left = getleft(nrows=100)
    X = esconnector.fit_transform(X=df_left)
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 3



