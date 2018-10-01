import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomClassifier(BaseEstimator, ClassifierMixin):
    """
    Create an estimator based on an ad-hoc function
    """

    def __init__(self, classificator):
        """

        Args:
            classificator (callable): function, shall return 0 or 1
        """
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)
        assert callable(classificator)
        self.classificator = classificator

    def fit(self, X, y, *args, **kwargs):
        return self

    def predict(self, X, *args, **kwargs):
        """

        Args:
            X (pd.DataFrame):
            *args:
            **kwargs:

        Returns:

        """
        assert isinstance(X, pd.DataFrame)
        score = X.apply(lambda row: self.classificator(row), axis=1)
        return score


class ThresholdBased(CustomClassifier):
    def __init__(self, on_cols, threshold=0.5, take='any', *args, **kwargs):
        """

        Args:
            on_cols: columns to use for the decision
            threshold (float):
            take (str): can be 'all' or 'any' --> any or all of the columns are above the threshold
            *args:
            **kwargs:
        """
        assert take in ['all', 'any']
        self.take = take
        self.on_cols = on_cols
        self.threshold = threshold
        if self.take == 'any':
            classificator = lambda row: int(any(map(lambda c: row[c] > self.threshold, self.on_cols)))
        elif self.take == 'all':
            classificator = lambda row: int(all(map(lambda c: row[c] > self.threshold, self.on_cols)))
        else:
            raise KeyError('take should be all or any')
        CustomClassifier.__init__(self, classificator=classificator, *args, **kwargs)
        pass
