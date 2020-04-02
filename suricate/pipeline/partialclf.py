import pandas as pd
from sklearn.base import ClassifierMixin

from suricate.preutils import concatixnames


class PartialClf(ClassifierMixin):
    def __init__(self,
                 classifier,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """
        This is a wrapper around a classifier that allows it to train on partial data
        where X and y do not have the same index, (because of pruning steps,...)
        It will train (fit) the classifier on the common index
        Args:
            classifier (ClassifierMixin): Classifier to use. Should be the output of the pipeline
            ixname (str):
            lsuffix (str):
            rsuffix (str):
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
        self.classifier = classifier
        pass

    def fit(self, X, y):
        """
        Fit the classifier on the common index beetween X and y
        Args:
            X (pd.DataFrame): similarity matrix
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true}

        Returns:
            self
        """
        ix_common = X.index.intersection(y.index)
        # Make sure the common index is not empty
        assert len(ix_common) > 0
        ## Make sure we have both classes in y_true in the common index
        assert len(y.loc[ix_common].unique()) >= 2
        self.classifier.fit(X=X.loc[ix_common], y=y.loc[ix_common])
        return self

    def predict(self, X):
        """
        Predict the matches, call predict on X
        Args:
            X (pd.DataFrame): similarity matrix
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true}

        Returns:
            pd.Series
        """
        return pd.Series(index=X.index, data=self.classifier.predict(X=X), name='y_pred')

    def predict_proba(self, X):
        """
        Predict the probability matches, call predict_proba on X
        Args:
            X (pd.DataFrame): similarity matrix
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true}

        Returns:
            pd.Series
        """
        return pd.Series(index=X.index, data=self.classifier.predict_proba(X=X), name='y_proba')

    def score(self, X, y, sampleweight=None):
        """
        Call the score function of the classifier on the common index
        Args:
            X:
            y:
            sampleweight:

        Returns:
            float
        """
        ix_common = X.index.intersection(y.index)
        return self.classifier.score(X=X.loc[ix_common], y=y.loc[ix_common], sample_weight=sampleweight)
