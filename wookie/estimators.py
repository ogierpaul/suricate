from sklearn.base import BaseEstimator


class CustomClassifier(BaseEstimator):
    """
    Create an estimator based on an ad-hoc function
    """

    def __init__(self, classificator_func):
        BaseEstimator.__init__(self)
        self.classificator_func = classificator_func
        assert callable(self.classificator_func)

    def fit(self, X, y, *args, **kwargs):
        return self

    def predict(self, X, *args, **kwargs):
        score = X.apply(lambda r: self.classificator_func(r), axis=1)
        score = score.apply(lambda r: r > 0.5)
        return score
