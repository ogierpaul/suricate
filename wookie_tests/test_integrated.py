from unittest import TestCase

import numpy as np
import pandas as pd

from wookie.comparators import PipeComparator
from wookie.connectors import Cartesian
from wookie_tests.db_builder import create_training_database, create_gid_database


class TestIntegrated(TestCase):
    def test_load_connect_data(self):
        print("\n******\n")
        # Load data
        left_data, right_data = create_gid_database()
        training_data = create_training_database()
        assert isinstance(training_data, pd.DataFrame)

        # Connect data
        con = Cartesian(reference=right_data, relevance_threshold=0.5)
        x_cart = con.fit_transform(left_data)
        assert isinstance(x_cart, pd.DataFrame)
        print("\n shape of cartesian data:\n {}".format(x_cart.shape[0]))
        score = x_cart[con.relevanceCol].apply(pd.Series)
        assert isinstance(score, pd.DataFrame)
        print("\n average score: \n {} ".format(score.mean(axis=1).mean()))

        # Compare data
        scoreplan = {
            'name': ['token'],
            'street': ['fuzzy'],
            'city': ['fuzzy']
        }
        pipe = PipeComparator(
            scoreplan=scoreplan
        )
        x_score = pipe.fit_transform(x_cart, n_jobs=2)
        assert isinstance(x_score, np.ndarray)
        print("\n******")
        pass
