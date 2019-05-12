import pytest
from fuzzywuzzy.fuzz import WRatio, QRatio, ratio, partial_token_sort_ratio

from wookie.comparators.fuzzy import simple_score, token_score


@pytest.fixture
def myscores():
    s = {
        'WRatio': WRatio,
        'QRatio': QRatio,
        'ratio': ratio,
        'partial_token_sort_ratio': partial_token_sort_ratio,
        'wookie_simple': simple_score,
        'wookie_token': token_score
    }
    return s


def test_simplescore(mymatches, myscores):
    df = mymatches.copy()
    for k in myscores.keys():
        df[k] = df.apply(lambda r: myscores[k](r['left'], r['right']), axis=1)
    print(df.transpose())
    assert True
