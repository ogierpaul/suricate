from sklearn.base import TransformerMixin, BaseEstimator
class PredtoTrans(TransformerMixin):
    def __init__(self, estimator):
        TransformerMixin.__init__(self)
        self.estimator = estimator
        pass

    def fit(self, X, y=None):
        self.estimator.fit(X=X, y=y)
        return self

    def transform(self, X):
        return self.estimator.predict(X=X)


class TranstoPred(BaseEstimator):
    def __init__(self, transformer):
        BaseEstimator.__init__(self)
        self.transformer = transformer
        pass

    def fit(self, X, y=None):
        self.transformer.fit(X=X, y=y)
        return self

    def predict(self, X):
        return self.transformer._transform(X=X)

    def fit_predict(self, X, y=None):
        return self.transformer.fit_transform(X=X, y=y)