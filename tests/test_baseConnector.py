from unittest import TestCase

import pandas as pd
from sklearn.base import TransformerMixin

from wookie.connectors import BaseConnector


class TestBaseConnector(TestCase):
    def test_transform(self):
        x = pd.DataFrame([['a', 'b'], ['c', 'd']])
        bc = BaseConnector()
        assert isinstance(bc, TransformerMixin)
        x1 = bc.transform(x)
        assert isinstance(x1, pd.DataFrame)

    def test_fit(self):
        x = pd.DataFrame([['a', 'b'], ['c', 'd']])
        bc = BaseConnector()
        bc.fit(x)
        assert isinstance(bc, TransformerMixin)
        assert isinstance(bc, BaseConnector)

    def test_pipeline(self):
        from sklearn.pipeline import make_union
        x = pd.DataFrame([['a', 'b'], ['c', 'd']])
        stages = [BaseConnector() for c in range(3)]
        pipe = make_union(*stages)
        pipe.fit_transform(x)
