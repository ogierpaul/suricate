# import pandas as pd
from sklearn.base import TransformerMixin


# This dont do anything, just pass the data as it is
class DataPasser(TransformerMixin):
    """
    This dont do anything, just pass the data as it is
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        This dont do anything, just pass the data as it is
        Args:
            X:

        Returns:

        """
        return X
