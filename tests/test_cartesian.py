from unittest import TestCase

import pandas as pd
from sklearn.base import TransformerMixin

from tests.db_builder import create_foo_database
from wookie.connectors import Cartesian, BaseConnector


class TestCartesian(TestCase):
    def test_transform(self):
        left, right = create_foo_database()
        transfo = Cartesian(
            right,
            relevance_threshold=0.1
        )
        assert isinstance(transfo, TransformerMixin)
        assert isinstance(transfo, BaseConnector)

        x_cart = transfo.transform(
            left
        )

        assert isinstance(x_cart, pd.DataFrame)
        print("\n result of transformation  \n", x_cart)
        pass

    def test_fit(self):
        left, right = create_foo_database()
        transfo = Cartesian(
            right,
            relevance_threshold=0.1
        )
        transfo.fit(left)
        pass

    def test_relevance_score(self):
        row = pd.Series(
            {
                'name_left': 'foo',
                'name_right': 'foo',
                'street_left': 'munich',
                'street_right': 'munchen'
            }
        )
        # create an empty transformator
        transfo = Cartesian(
            pd.DataFrame()
        )
        score = transfo.relevance_score(row)
        pass

    def test_pruning(self):
        transfo = Cartesian(
            pd.DataFrame(),
            relevance_threshold=0.0
        )
        relevance1 = {
            'name': 1.0,
            'street': 0.0
        }
        score = transfo.pruning(relevance1)
        assert score is True
        relevance2 = {
            'name': 0.0,
            'street': 0.0
        }
        score = transfo.pruning(relevance2)
        assert score is False

        pass
