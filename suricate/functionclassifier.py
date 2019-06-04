from sklearn.base import ClassifierMixin


class FunctionClassifier(ClassifierMixin):
    """
    This class is a classifier that is pre-programmed.
    The func should return a boolean or a classifier (0, 1)
    """

    def __init__(self, func):
        ClassifierMixin.__init__(self)
        self.func = func
        assert callable(self.func)

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        y = self.func(X)
        y = y.astype(int)
        return y
