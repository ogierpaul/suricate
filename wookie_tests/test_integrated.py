from unittest import TestCase

import pandas as pd

n_jobs = 2
from wookie.comparators import PipeComparator
from wookie.connectors import Cartesian
from wookie_tests.db_builder import create_training_database, create_gid_database
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


class TestIntegrated(TestCase):
    def test_load_connect_data(self):
        print("\n******\n")
        # Load data
        left_data, right_data = create_gid_database()
        x_train, y_train = create_training_database()
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)

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

        scoring_pipe = PipeComparator(
            scoreplan=scoreplan
        )
        rf = RandomForestClassifier(n_jobs=n_jobs, n_estimators=50, max_depth=10)

        pipe = make_pipeline(*[scoring_pipe, rf])
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_cart)

        show_results = x_cart.loc[y_pred.astype(bool)]
        print("\n", show_results)

        print("\n******")
        pass
