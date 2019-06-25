import pytest
from fuzzywuzzy.fuzz import WRatio, QRatio, ratio, partial_token_sort_ratio

from suricate.sbsdftransformers.funcsbscomparator import simple_score, token_score
from suricate.data.circus import getXsbs
X_sbs = getXsbs()

# from ..data.circus import mymatches


myscores = {
        'WRatio': WRatio,
        'QRatio': QRatio,
        'ratio': ratio,
        'partial_token_sort_ratio': partial_token_sort_ratio,
        'wookie_simple': simple_score,
        'wookie_token': token_score
    }


def test_simplescore():
    df = X_sbs.copy()
    for k in myscores.keys():
        df[k] = df.apply(lambda r: myscores[k](r['name_left'], r['name_right']), axis=1)
    print(df.transpose())
    assert True
