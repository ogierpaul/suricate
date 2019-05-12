import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, ClassifierMixin

from wookie.preutils import concatixnames, createmultiindex, addsuffix


class LrModel(ClassifierMixin):
    def __init__(self,
                 scorer,
                 classifier,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """

        Args:
            scorer (TransformerMixin):
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
        self.scorer = scorer
        self.classifier = classifier
        pass

    def fit(self, X, y):
        '''
        Fit the transformer
        Args:
            X (list): list of [df_left, df_right]
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true}

        Returns:
            self
        '''
        X_score = self.scorer.fit_transform(X=X, y=None)
        X_slice = self.reindex(X=X, X_score=X_score, y=y)
        self.classifier.fit(X=X_slice, y=y)
        return self

    def reindex(self, X, X_score, y=None):
        '''
        Transform X_score, output of X through the score,  into X_slice, sliced according to y_true (pd.Series)
        X [df_left, df_right] -[scorer]-> X_score -[reindex]-> X_score.loc[y_true.index]
        Into
        Args:
            X (list) : is a list containing (df_left, df_right)
            X_score (np.ndarray): X is a numpy.ndarray which is a cartesian product of df_left and df_right
            y (pd.Series/np.ndarray): y is either: /
                - pd.Series containing the supervised scores: pairs {['ix_left', 'ix_right']: y_true} which can be a slice of x
                - numpy.ndarray of [0,1 ,0 ,1 ...] which must be same length as x
                - None --> return X

        Returns:
            np.ndarray: Slice of X_score
        '''
        self.all_ix = createmultiindex(X=X, names=self.ixnamepairs)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            return pd.DataFrame(data=X_score, index=self.all_ix).loc[y.index]
        elif isinstance(y, np.ndarray):
            if len(y) != X_score.shape[0]:
                raise IndexError(
                    'expected length of numpy.ndarray: ({}), given length of y: {}'.format(X.shape[0], len(y)))
            else:
                return X
        elif y is None:
            return X
        else:
            return X

    def predict(self, X):
        X_score = self.scorer.transform(X=X)
        return self.classifier.predict(X=X_score)

    def predict_proba(self, X):
        X_score = self.scorer.transform(X=X)
        return self.classifier.predict_proba(X=X_score)

    def score(self, X, y, sampleweight=None):
        X_score = self.scorer.transform(X=X)
        X_slice = self.reindex(X=X, X_score=X_score, y=y)
        return self.classifier.score(X=X_slice, y=y, sample_weight=sampleweight)

    def return_pairs(self, X):
        return pd.Series(
            index=createmultiindex(X=X, names=self.ixnamepairs),
            data=self.predict(X)
        )

    def show_pairs(self, X, y=None, use_cols=None):
        """
        Create a side by side table from a list of pairs (as a DataFrame)
        Args:
            X
            y (pd.DataFrame/pd.Series): of the form {['ix_left', 'ix_right']:['y_true']}
            use_cols (list): columns to use

        Returns:
            pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
        """
        left = X[0]
        right = X[1]

        if y is None:
            xpairs = pd.DataFrame(index=createmultiindex(X=X, names=self.ixnamepairs))
        elif isinstance(y, pd.DataFrame):
            xpairs = y.copy()
        else:
            assert isinstance(y, pd.Series)
            xpairs = pd.DataFrame(y.copy())

        xpairs = xpairs.reset_index(drop=False)

        if use_cols is None or len(use_cols) == 0:
            use_cols = left.columns.intersection(right.columns)
        xleft = left[use_cols].copy().reset_index(drop=False)
        xright = right[use_cols].copy().reset_index(drop=False)
        xleft = addsuffix(xleft, self.lsuffix).set_index(self.ixnameleft)
        xright = addsuffix(xright, self.rsuffix).set_index(self.ixnameright)

        sbs = xpairs.join(
            xleft, on=self.ixnameleft, how='left'
        ).join(
            xright, on=self.ixnameright, how='left'
        ).set_index(
            self.ixnamepairs
        )
        return sbs
