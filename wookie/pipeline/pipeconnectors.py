from sklearn.base import TransformerMixin, ClassifierMixin


def reindex_on_y(X, y):
    # TODO
    X2 = X
    return X2


class ReindexClassifier(ClassifierMixin):
    def __init__(self, clf):
        ClassifierMixin.__init__(self)
        self.clf = clf

    def fit(self, X, y):
        X2 = reindex_on_y(X=X, y=y)
        self.clf.fit(X=X2, y=y)
        return self

    def predict(self, X):
        self.clf.predict(X=X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        X2 = reindex_on_y(X=X, y=y)
        return self.clf.score(X=X, y=y, sample_weight=sample_weight)


class PruningClf(ReindexClassifier):
    def __init__(self, clf):
        ReindexClassifier.__init__(self, clf=clf)
        pass

    def predict(self, X):
        y = self.clf.predict(X)
        y = y.loc[y > 1]
        X = X.loc[y]
        return X


class PruningPandasDF(TransformerMixin):
    def __init__(self, usefunc):
        TransformerMixin.__init__(self)
        self.usefunc = usefunc
        # usefunc: takes X as input, return vector of boolean
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        score = X.apply(lambda r: self.usefunc(r))
        return X.loc[score]
