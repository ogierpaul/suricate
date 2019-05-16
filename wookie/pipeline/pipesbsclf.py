import pandas as pd
from sklearn.base import TransformerMixin, ClassifierMixin

from wookie.preutils import concatixnames


class PipeSbsClf(ClassifierMixin):
    def __init__(self,
                 transformer,
                 classifier,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """

        Args:
            transformer (TransformerMixin):
            classifier (ClassifierMixin):
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            n_jobs (int):
            pruning_ths (float): return only the pairs which have a score greater than the store_ths
        """
        ClassifierMixin.__init__(self)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        self.fitted = False
        self.transformer = transformer
        self.classifier = classifier
        pass

    def fit(self, X, y):
        '''
        Fit the transformer
        Args:
            X (pd.DataFrame): side by side [name_left; name_right, ...]
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true}

        Returns:
            self
        '''
        X_score = self.transformer.fit_transform(X=X, y=None)
        self.classifier.fit(X=X_score, y=y)
        return self

    def predict(self, X):
        X_score = self.transformer.transform(X=X)
        return self.classifier.predict(X=X_score)

    def predict_proba(self, X):
        X_score = self.transformer.transform(X=X)
        return self.classifier.predict_proba(X=X_score)

    def score(self, X, y, sampleweight=None):
        X_score = self.transformer.transform(X=X)
        return self.classifier.score(X=X_score, y=y, sample_weight=sampleweight)
