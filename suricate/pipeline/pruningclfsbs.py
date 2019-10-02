import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, TransformerMixin, ClusterMixin
from sklearn.metrics.classification import accuracy_score
from suricate.preutils import concatixnames, createmultiindex
from suricate.explore import ClusterClassifier


class PruningClfSbs(ClassifierMixin):
    def __init__(self,
                 lr_scorer,
                 lr_connector,
                 cluster,
                 clusterclassifier,
                 sbsmodel,
                 classifier,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """

        Args:
            lr_scorer (TransfomerMixin): Scorer used to do the calculation,
            lr_connector (TransformerMixin: Connector used to create a Sbs view of the data
            cluster (ClusterMixin): Cluster used to predict
            sbsmodel (PipeSbsClf): SbSModel
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
        self.lr_scorer = lr_scorer
        self.lr_connector = lr_connector
        self.cluster = cluster
        self.clusterclassifier = clusterclassifier
        self.sbsmodel = sbsmodel
        self.classifier = classifier
        pass

    def fit(self, X, y_lr=None, y_sbs=None):
        """
        Fit the transformer
        Args:
            X (pd.DataFrame): side by side [name_left; name_right, ...]
            y_lr (pd.Series): pairs {['ix_left', 'ix_right']: y_true} for the pruning model
            y_sbs (pd.Series): pairs {['ix_left', 'ix_right']: y_true} for the predictive model

        Returns:
            self
        """
        return self._pipe(X=X, y_lr=y_lr, y_sbs=y_sbs, fit=True)

    def _pipe(self, X, y_true, fit=False, proba=False):
        """
        # select only positive matches y_pred_lr == 1.0 from first classifier for further scoring
        # Add as well as sure matches y_pred_lr == 2.0
        Args:
            X (list): [df_left, df_right]
            y_lr (pd.Series): training data for the first model (LrModel)
            y_sbs (pd.Series): training data for the second model (SbsModel)
            fit (bool): True: fit all transformers / classifiers and return self. False: return y_pred
            proba (bool): Only works if fit is False. If fit is False and proba is True: return y_proba. If fit if False and proba is False: return y_pred

        Returns:
            array
        """

        # Fit the first model
        if fit is True:
            self.lr_scorer.fit(X=X)

        # Get the first score
        X_lr_score = self.lr_scorer.getscore(X=X)
        ix_lr = self.lr_scorer.getindex(X=X)
        Xsbs = self.lr_scorer.getsbs(X=X)

        # If fit is true, slice this score according to scope of y_lr
        if fit is True:
            X_lr_score, y_lr_slice, ix_slice = self.lrmodel.slice(X=X, X_score=X_lr_score, y=y_lr)
            ix_lr = ix_slice
        else:
            ix_lr = createmultiindex(X=X, names=self.ixnamepairs)

        # Get the prediction for the X_lrscope
        y_lr_pred = pd.Series(
            data=self.lrmodel.classifier.predict(X=X_lr_score),
            index=ix_lr,
            name='y_lr_pred'
        )
        # select only positive matches from first classifier
        ix_lr_pos = y_lr_pred.loc[y_lr_pred == 1].index

        # intersect this index with the ones
        if fit is True:
            ix_sbs = ix_lr_pos.intersection(y_sbs.index)
        else:
            ix_sbs = ix_lr_pos

        X_lr_score = pd.DataFrame(data=X_lr_score, index=ix_lr).loc[ix_sbs]

        # Create the input dataframe needed for the SbsModel
        X_Sbs = cartesian_join(
            left=X[0],
            right=X[1],
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            on_ix=ix_sbs
        )
        # And Transform (Second scoring engine)
        if fit is True:
            self.sbsmodel.transformer.fit(X=X_Sbs)
        X_sbs_score = self.sbsmodel.transformer.transform(X=X_Sbs)

        # Merge the output of the two scores
        X_final_score = np.hstack((X_lr_score.values, X_sbs_score))

        if fit is True:
            self.sbsmodel.classifier.fit(X=X_final_score, y=y_sbs.loc[ix_sbs])
            return self
        else:
            # If we are not for fit we are for pred
            if proba is True:
                y_pred = self.sbsmodel.classifier.predict_proba(X=X_final_score)
            else:
                y_pred = self.sbsmodel.classifier.predict(X=X_final_score)
            y_pred_all = pd.Series(index=createmultiindex(X=X, names=self.ixnamepairs)).fillna(0)
            y_pred_all.loc[ix_sbs] = y_pred
            y_pred_all.loc[y_lr_pred == 2.0] = 1.0
            return y_pred_all

    def predict(self, X):
        """
        # select only positive matches y_pred_lr == 1.0 from first classifier for further scoring
        # Add as well as sure matches from y_pred_lr == 2.0 (Case of clusterer for example)
        Args:
            X: [df_left, df_right]

        Returns:
            np.ndarray
        """

        return self._pipe(X=X, fit=False)

    def predict_proba(self, X):
        return self._pipe(X=X, fit=False, proba=True)

    def score(self, X, y, sampleweight=None):
        y_pred = pd.Series(
            data=self.predict(X=X),
            index=createmultiindex(X=X, names=self.ixnamepairs),
            name='y_pred'
        )
        ix_common = y.index.intersection(y_pred.index)
        return accuracy_score(
            y_pred=y_pred.loc[ix_common],
            y_true=y.loc[ix_common],
            sample_weight=sampleweight
        )