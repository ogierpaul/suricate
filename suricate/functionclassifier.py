from sklearn.base import ClassifierMixin
import numpy as np

class FunctionClassifier(ClassifierMixin):
    """
    This class is a classifier that is pre-programmed.
    The func should return an integer
    """

    def __init__(self, func):
        ClassifierMixin.__init__(self)
        self.func = func
        assert callable(self.func)

    def fit(self, X=None, y=None):
        """

        Args:
            X (np.array): of shape (n_samples, n_features)
            y (np.array): of shape (n_samples, 1)

        Returns:
            FunctionClassifier
        """
        return self

    def predict(self, X):
        """

        Args:
            X (np.array): of shape (n_samples, n_features)

        Returns:
            np.array: of shape (n_samples, 1) as integer
        """
        y = self.func(X)
        y = y.astype(int)
        return y
