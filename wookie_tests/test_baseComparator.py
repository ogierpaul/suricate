from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from wookie.comparators import BaseComparator
from wookie.scoreutils import exact_score


class TestBaseComparator(TestCase):
    def test_transform(self):
        x = pd.DataFrame(
            {
                'name_left': ['foo', 'bar'],
                'name_right': ['foo', 'baz']
            }
        )
        bc = BaseComparator(
            left='name_left',
            right='name_right',
            compfunc=exact_score
        )
        assert isinstance(bc, TransformerMixin)
        x1 = bc.transform(x)
        assert isinstance(x1, np.ndarray)
        total = x1.sum().sum()
        assert isinstance(total, float)
        pass

    def test_fit(self):
        x = pd.DataFrame(
            {
                'name_left': ['foo', 'bar'],
                'name_right': ['foo', 'baz']
            }
        )
        bc = BaseComparator(
            left='name_left',
            right='name_right',
            compfunc=exact_score
        )
        assert isinstance(bc, TransformerMixin)
        bc.fit(x)
        pass
