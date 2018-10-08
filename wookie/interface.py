import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics.classification import accuracy_score
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing.imputation import Imputer

from wookie import sbscomparators, connectors, oaacomparators


class SbsModel:
    def __init__(self, prefunc, idtfidf, sbscomparator, estimator):
        self.prefunc = prefunc
        assert isinstance(idtfidf, oaacomparators.IdTfIdfConnector)
        self.idtfidf = idtfidf
        self.sbscomparator = sbscomparator
        self.estimator = estimator
        self.skmodel = BaseEstimator()
        pass

    def fit(self, left, right, pairs):
        """

        Args:
            left (pd.DataFrame): {'ixname': [cols..] }
            right (pd.DataFrame): {'ixname': [cols..] }
            pairs (pd.DataFrame): {['ixname_left', 'ixname_right']: [y_true] }

        Returns:

        """
        newleft = self.prefunc(left)
        newright = self.prefunc(right)
        connectscores = self.idtfidf.transform(newleft, newright)
        ixtrain = pairs.index.intersection(connectscores.index)
        X_sbs = connectors.createsbs(pairs=connectscores, left=newleft, right=newright)
        X_train = X_sbs.loc[ixtrain]
        y_train = pairs.loc[ixtrain, 'y_true']
        dp_connectscores = sbscomparators.DataPasser(on_cols=connectscores.columns.tolist())
        # noinspection PyTypeChecker
        scorer = make_union(
            *[
                self.sbscomparator,
                dp_connectscores
            ]
        )
        ## Estimator
        imp = Imputer()
        self.skmodel = make_pipeline(
            *[
                scorer,
                imp,
                self.estimator
            ]
        )
        self.skmodel.fit(X_train, y_train)
        pass

    def predict(self, left, right):
        newleft = self.prefunc(left)
        newright = self.prefunc(right)
        connectscores = self.idtfidf.transform(newleft, newright)
        X_sbs = connectors.createsbs(pairs=connectscores, left=newleft, right=newright)
        y_pred = self.skmodel.predict(X_sbs)
        y_pred = pd.Series(y_pred, index=X_sbs.index)
        return y_pred

    def score(self, left, right, pairs):
        y_pred = self.predict(left=left, right=right)
        assert isinstance(y_pred, pd.Series)
        y_true = pairs['y_true']
        assert isinstance(y_true, pd.Series)
        y_pred2 = pd.Series(index=y_true.index)
        y_pred2.loc[y_true.index.intersection(y_pred.index)] = y_pred
        y_pred2.fillna(0, inplace=True)
        score = accuracy_score(y_true=y_true, y_pred=y_pred2)
        return score

# class ColumnBased:
#     def __init__(self, estimator, prefunc=None, id_cols=None, free_cols=None, stop_words=None):
#         """
#         Args:
#             estimator (BaseEstimator): Sklearn Estimator
#             prefunc (callable):
#             id_cols (list):
#             free_cols (list):
#             stop_words (dict):
#
#         """
#         assert issubclass(type(estimator), BaseEstimator)
#
#         if prefunc is not None:
#             assert isinstance(prefunc, callable)
#         self.prefunc = prefunc
#         self.scoreplan = dict()
#         if id_cols is not None:
#             assert hasattr(id_cols, '__iter__')
#             for c in id_cols:
#                 self.scoreplan[c] = list()
#                 self.scoreplan[c] = ['fuzzy', 'exact', 'token']
#         self.id_cols = id_cols
#         if free_cols is not None:
#             assert hasattr(free_cols, '__iter__')
#             for c in free_cols:
#                 self.scoreplan[c] = ['exact']
#         self.free_cols = free_cols
#         if stop_words is not None:
#             assert isinstance(stop_words, dict)
#         self.stop_words = stop_words
#         self.tokenizers = dict()
#         scoreplan = dict()
#         if self.free_cols is not None:
#             for c in self.free_cols:
#                 scoreplan[c] = list()
#             scoreplan[c].append('wostopwords')
#         self.sbsscores = sbscomparators.PipeSbsComparator(
#             scoreplan={
#                 # 'name_ascii': ['exact', 'fuzzy', 'token'],
#                 # 'street_ascii': ['exact', 'token'],
#                 # 'street_ascii_wostopwords': ['token'],
#                 # 'name_ascii_wostopwords': ['fuzzy'],
#                 # 'city': ['fuzzy'],
#                 # 'postalcode_ascii': ['exact'],
#                 # 'postalcode_2char': ['exact'],
#                 'countrycode': ['exact']
#             }
