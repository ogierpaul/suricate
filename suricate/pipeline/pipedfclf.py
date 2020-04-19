import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, ClassifierMixin

from suricate.preutils import concatixnames, createmultiindex, addsuffix

# THIS SHOULD BE OBSOLETE
class PipeDfClf(ClassifierMixin):
    def __init__(self,
                 transformer,
                 classifier,
                 ixname='ix',
                 source_suffix='source',
                 target_suffix='target',
                 **kwargs):
        """

        Args:
            transformer (TransformerMixin): Transformer --> CLF
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
            X (list): list of [df_source, df_target]
            y (pd.Series): pairs {['ix_source', 'ix_target']: y_true}

        Returns:
            self
        """
        X_score = self.transformer.fit_transform(X=X, y=None)
        X_slice, y_slice, ix_slice = self.slice(X=X, X_score=X_score, y=y)
        self.classifier.fit(X=pd.DataFrame(X_slice, index=ix_slice), y=pd.Series(y_slice, index=ix_slice))
        return self

    def slice(self, X, X_score, y=None):
        """
        Transform X_score, output of X through the score,  into X_slice, sliced according to y_true (pd.Series)
        X [df_source, df_target] -[scorer]-> X_score -[reindex]-> X_score.loc[y_true.index]
        Into
        Args:
            X (list) : is a list containing (df_source, df_target)
            X_score (np.ndarray): X is a numpy.ndarray which is a cartesian product of df_source and df_target
            y (pd.Series/np.ndarray): y is either: /
                - pd.Series containing the supervised scores: pairs {['ix_source', 'ix_target']: y_true} which can be a slice of x
                - numpy.ndarray of [0,1 ,0 ,1 ...] which must be same length as x
                - None --> return X

        Returns:
            np.ndarray, np.ndarray, pd.Index: Slice of X_score, y, common index
        """
        ix_all = createmultiindex(X=X, names=self.ixnamepairs)
        X_score = pd.DataFrame(X_score, index=ix_all)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            commonindex = X_score.index.intersection(y.index)
            return X_score.loc[commonindex].values, y.loc[commonindex].values, commonindex
        elif y is None:
            return X_score.values, None, ix_all
        else:
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
            y (pd.DataFrame/pd.Series): of the form {['ix_source', 'ix_target']:['y_true']}
            use_cols (list): columns to use

        Returns:
            pd.DataFrame {['ix_source', 'ix_target'] : ['name_source', 'name_target', .....]}
        """
        source = X[0]
        target = X[1]

        if y is None:
            xpairs = pd.DataFrame(index=createmultiindex(X=X, names=self.ixnamepairs))
        elif isinstance(y, pd.DataFrame):
            xpairs = y.copy()
        else:
            assert isinstance(y, pd.Series)
            xpairs = pd.DataFrame(y.copy())

        xpairs = xpairs.reset_index(drop=False)

        if use_cols is None or len(use_cols) == 0:
            use_cols = source.columns.intersection(target.columns)
        xsource = source[use_cols].copy().reset_index(drop=False)
        xright = target[use_cols].copy().reset_index(drop=False)
        xsource = addsuffix(xsource, self.source_suffix).set_index(self.ixnamesource)
        xright = addsuffix(xright, self.target_suffix).set_index(self.ixnametarget)

        sbs = xpairs.join(
            xsource, on=self.ixnamesource, how='left'
        ).join(
            xright, on=self.ixnametarget, how='left'
        ).set_index(
            self.ixnamepairs
        )
        return sbs
