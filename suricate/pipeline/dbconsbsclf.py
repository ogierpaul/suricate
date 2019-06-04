from sklearn.base import ClassifierMixin

from suricate.pipeline.pipesbsclf import PipeSbsClf
from suricate.preutils import concatixnames


class DbconSbsClf(ClassifierMixin):
    def __init__(self,
                 transformer,
                 classifier,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """

        Args:
            transformer : connect and return the results for the left side
            classifier (PipeSbsClf)
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
        """
            Args:
            X(pd.DataFrame): "left" dataframe
            y (pd.Series):

        Returns:

        """
        # The self.transformer must return a dataframe of the form {[ix_left, ix_right]:[name_left, name_right, score]}
        X_sbs = self.transformer.transform(X)
        on_ix = y.index.intersection(X_sbs.index)
        self.classifier.fit(X=X_sbs.loc[on_ix], y=y.loc[on_ix])
        return self

    def predict(self, X):
        X_sbs = self.transformer.transform(X)
        return self.classifier.predict(X=X_sbs)

    def predict_proba(self, X):
        X_sbs = self.transformer.transform(X)
        return self.classifier.predict_proba(X=X_sbs)

    def score(self, X, y, sampleweight=None):
        X_sbs = self.transformer.transform(X=X)
        return self.classifier.score(X=X_sbs, y=y)
