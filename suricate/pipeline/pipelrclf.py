import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, ClassifierMixin

from suricate.preutils import concatixnames, createmultiindex, addsuffix


class PipeLrClf(ClassifierMixin):
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
            X (list): list of [df_left, df_right]
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true}

        Returns:
            self
        '''
        X_score = self.transformer.fit_transform(X=X, y=None)
        X_slice, y_slice, ix_slice = self.slice(X=X, X_score=X_score, y=y)
        self.classifier.fit(X=X_slice, y=y_slice)
        return self

    def slice(self, X, X_score, y=None):
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
            np.ndarray, np.ndarray: Slice of X_score, y
        '''
        ix_all = createmultiindex(X=X, names=self.ixnamepairs)
        X_score = pd.DataFrame(X_score, index=ix_all)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            commonindex = X_score.index.intersection(y.index)
            return X_score.loc[commonindex].values, y.loc[commonindex].values, commonindex
        elif y is None:
            return X_score.values, None, ix_all
        else:
            # TODO: Prio 2: Could be used to treat case of y as masked array
            return X_score.values, y, ix_all

    def predict(self, X):
        X_score = self.transformer.transform(X=X)
        return self.classifier.predict(X=X_score)

    def fit_predict(self, X, y):
        self.fit(X=X, y=y)
        y_pred = self.predict(X=X)
        return y_pred

    def predict_proba(self, X):
        X_score = self.transformer.transform(X=X)
        return self.classifier.predict_proba(X=X_score)

    def score(self, X, y, sampleweight=None):
        X_score = self.transformer.transform(X=X)
        X_slice, y_slice, ix_slice = self.slice(X=X, X_score=X_score, y=y)
        return self.classifier.score(X=X_slice, y=y_slice, sample_weight=sampleweight)

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
