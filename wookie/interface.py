from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing.imputation import Imputer

from wookie import sbscomparators, connectors


class SbsModel:
    def __init__(self, pairs, left, right, prefunc, idtfidf, sbscomparator, est):
        self.pairs = pairs
        self.left = left
        # self.newleft = self.left
        # self.newright = self.right
        self.right = right
        self.prefunc = prefunc
        self.idtfidf = idtfidf
        self.sbscomparator = sbscomparator
        self.estimator = est
        pass

    def fit(self, left, right, pairs):
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
        model = make_pipeline(
            *[
                scorer,
                imp,
                self.estimator
            ]
        )
        model.fit(X_train, y_train)
        pass

    def transform(self, left, right):
        newleft = self.prefunc(left)
        newright = self.prefunc(right)
        connectscores = self.idtfidf.transform(newleft, newright)
        X_sbs = connectors.createsbs(pairs=connectscores.index, left=newleft, right=newright)
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
        model = make_pipeline(
            *[
                scorer,
                imp,
                self.estimator
            ]
        )
        y_pred = model.predict(X_sbs)
        return y_pred
