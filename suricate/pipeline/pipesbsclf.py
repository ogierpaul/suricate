import pandas as pd
from sklearn.base import TransformerMixin, ClassifierMixin

from suricate.preutils import concatixnames


class PipeSbsClf(ClassifierMixin):
    def __init__(self,
                 transformer,
                 classifier,
                 ixname='ix',
                 source_suffix='source',
                 target_suffix='target',
                 **kwargs):
        """

        Args:
            transformer (TransformerMixin):
            classifier (ClassifierMixin):
            ixname (str):
            source_suffix (str):
            target_suffix (str):
            n_jobs (int):
            pruning_ths (float): return only the pairs which have a score greater than the store_ths
        """
        ClassifierMixin.__init__(self)
        self.ixname = ixname
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix
        )
        self.fitted = False
        self.transformer = transformer
        self.classifier = classifier
        pass

    def fit(self, X, y):
        """
        Fit the transformer
        Args:
            X (pd.DataFrame): side by side [name_source; name_target, ...]
            y (pd.Series): pairs {['ix_source', 'ix_target']: y_true}

        Returns:
            self
        """
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
