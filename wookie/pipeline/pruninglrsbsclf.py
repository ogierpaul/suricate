import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from wookie.lrdftransformers import cartesian_join
from wookie.pipeline.pipelrclf import PipeLrClf
from wookie.pipeline.pipesbsclf import PipeSbsClf
from wookie.preutils import concatixnames, createmultiindex


class PruningLrSbsClf(ClassifierMixin):
    def __init__(self,
                 lrmodel,
                 sbsmodel,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """

        Args:
            lrmodel (PipeLrClf): LrModel
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
        self.lrmodel = lrmodel
        self.sbsmodel = sbsmodel
        pass

    def fit(self, X, y_lr=None, y_sbs=None):
        '''
        Fit the transformer
        Args:
            X (pd.DataFrame): side by side [name_left; name_right, ...]
            y_lr (pd.Series): pairs {['ix_left', 'ix_right']: y_true} for the pruning model
            y_sbs (pd.Series): pairs {['ix_left', 'ix_right']: y_true} for the predictive model

        Returns:
            self
        '''
        return self._pipe(X=X, y_lr=y_lr, y_sbs=y_sbs, fit=True)

    def _pipe(self, X, y_lr=None, y_sbs=None, fit=False, proba=False):
        """

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
            self.lrmodel.fit(X=X, y=y_lr)
        # Transform into scores the input with the scoring engine from the lrmodel
        X_lr_score = self.lrmodel.transformer.transform(X=X)

        # Calculate the index used for the SbsModel, second phase of scoring
        # To be index_consistent: if we are in fit phase take y_sbs to fit second (Sbs) Model
        # Else take the positive results (pairs) of the first classifier (Pruning)
        if fit is True:
            on_ix = y_sbs.index
        else:
            y_lr_pred = pd.Series(
                index=createmultiindex(X=X, names=self.ixnamepairs),
                data=self.lrmodel.classifier.predict(X=X_lr_score)
            )
            on_ix = y_lr_pred[y_lr_pred == 1].index
        # In addition, save the score of the lr model for the index used
        X_lr_score = pd.DataFrame(
            index=createmultiindex(X=X, names=self.ixnamepairs),
            data=X_lr_score
        ).loc[
            on_ix
        ].values

        # Create the input dataframe needed for the SbsModel
        X_Sbs = cartesian_join(
            left=X[0],
            right=X[1],
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            on_ix=on_ix
        )
        # And Transform (Second scoring engine)
        if fit is True:
            self.sbsmodel.transformer.fit(X=X_Sbs)
        X_sbs_score = self.sbsmodel.transformer.transform(X=X_Sbs)

        # Merge the output of the two scores
        X_final_score = np.hstack((X_lr_score, X_sbs_score))

        if fit is True:
            self.sbsmodel.classifier.fit(X=X_final_score, y=y_sbs)
            return self
        else:
            # If we are not for fit we are for pred
            if proba is True:
                y_pred = self.sbsmodel.classifier.predict_proba(X=X_final_score)
            else:
                y_pred = self.sbsmodel.classifier.predict(X=X_final_score)
            y_pred_all = pd.Series(index=createmultiindex(X=X, names=self.ixnamepairs)).fillna(0)
            y_pred_all.loc[on_ix] = y_pred
            return y_pred_all

    def predict(self, X):
        return self._pipe(X=X, fit=True)

    def predict_proba(self, X):
        return self._pipe(X=X, fit=True, proba=True)
