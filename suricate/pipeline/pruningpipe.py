import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, TransformerMixin, ClusterMixin
from sklearn.metrics.classification import accuracy_score
from suricate.preutils import concatixnames, createmultiindex
from suricate.explore import Explorer

#TODO: Review the doc
#TODO: Review the score method

class PruningPipe(ClassifierMixin):
    def __init__(self,
                 connector,
                 explorer,
                 sbsmodel,
                 classifier,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 **kwargs):
        """

        Args:
            connector (TransfomerMixin): Lr Df Connector (Scorer) used to do the calculation,
            explorer (Explorer): Classifier used to do the pruning (0=no match, 1: potential match, 2: sure match)
            sbsmodel (TransformerMixin): Side-by-Side scorer
            classifier (ClassifierMixin): Classifier used to do the prediction
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
        self.connector = connector
        self.explorer = explorer
        self.sbsmodel = sbsmodel
        self.classifier = classifier
        pass

    def fit(self, X, y):
        """
        Fit the transformer
        Args:
            X (pd.DataFrame): side by side [name_left; name_right, ...]
            y (pd.Series): pairs {['ix_left', 'ix_right']: y_true} for the training

        Returns:
            self
        """
        return self._pipe(X=X, y_true=y, fit=True)

    def _pipe(self, X, y_true, fit=False, proba=False):
        """
        # select only positive matches y_pred_lr == 1.0 from first classifier for further scoring
        # Add as well as sure matches y_pred_lr == 2.0
        Args:
            X: input data to the connector
            y_lr (pd.Series): training data for the first model (LrModel)
            y_sbs (pd.Series): training data for the second model (SbsModel)
            fit (bool): True: fit all transformers / classifiers and return self. False: return y_pred
            proba (bool): Only works if fit is False. If fit is False and proba is True: return y_proba. If fit if False and proba is False: return y_pred

        Returns:
            array
        """
        # First model: Connector
        ## Fit the first model
        if fit is True:
            self.connector.fit(X=X)

        ## Get the first score
        Xtc = self.connector.transform(X=X) # score matrix
        ixc = self.connector.getindex(X=X) # index of Xc

        # Second model: Explorer
        ## Fit the explorer
        if fit is True:
            self.explorer.fit(X=pd.DataFrame(data=Xtc, index=ixc), y=y_true, fit_cluster=True)
        ## Get the pruning classifier
        y_pruning = pd.Series(
            data=self.explorer.predict(X=Xtc),
            index=ixc
        )

        ## Get the mixed matches from the pruning classifier
        ### Save the sure matches
        ix_sure = y_pruning.loc[y_pruning == 2].index
        ## Save the sure negative matches
        ix_neg = y_pruning.loc[y_pruning == 0].index
        # select only possible (mixed) matches from first classifier
        ix_mix = y_pruning.loc[y_pruning == 1].index
        Xs_mix = self.connector.getsbs(X=X, on_ix=ix_mix) # side by side view of ix_mix
        Xtc_mix = pd.DataFrame(data=Xtc, index=ixc).loc[ix_mix]
        # And Transform (Second scoring engine)
        if fit is True:
            self.sbsmodel.fit(X=Xs_mix)
        Xts_mix = self.sbsmodel.transform(X=Xs_mix)

        # Merge the output of the two scores
        Xtf = np.hstack((Xtc_mix.values, Xts_mix.values))

        if fit is True:
            # select only the intersection of y_true and ix_mix:
            ix_train = ix_mix.intersection(y_true.index)
            assert len(ix_train) > 0
            Xtf_train= pd.DataFrame(data=Xtf, index=ix_mix).loc[ix_train]
            y_true_train = y_true.loc[ix_train]
            assert y_true_train.value_counts().shape[0] == 2
            self.classifier.fit(X=Xtf_train, y=y_true_train)
            return self
        else:
            # If we are not for fit we are for pred
            if proba is True:
                y_pred = self.classifier.predict_proba(X=Xtf)
            else:
                y_pred = self.classifier.predict(X=Xtf)

            # Format the results
            ## Create a series to contain the results
            y_pred_all = pd.Series(index=ixc, name='y_pred')
            ## Results where pruning gives possible (mixed) results are given y_pred (classification or probability from classifier)
            y_pred_all.loc[ix_mix] = y_pred
            y_pred_all.loc[ix_neg] = 0
            y_pred_all.loc[ix_sure] = 1.0
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