import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing.imputation import Imputer

from wookie import sbscomparators, connectors, oaacomparators

_tfidf_threshold_value = 0.3


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


class ColumnBased:
    """
            score plan
        {
            name_col:{
                'type': one of ['FreeText', 'Category', 'Id', 'Code']
                'stop_words': [list of stopwords]
                'score': (defaults)
                    for free text:
                        ['exact', 'fuzzy', 'tf-idf', 'token_wostopwords']
                    for category:
                        ['exact'] --> but not taken into account for pruning
                    for Id:
                        ['exact']
                    for code: --> but not taken into account for pruning
                        ['exact', 'first n_digits/ngrams' --> to do]
                'threshold'
                    'tf-idf' --> Default 0.3
                    'exact' --> Default 1
            }
        }
    """

    def __init__(self, scoreplan, prefunc=None, estimator=None, ixname='ix', lsuffix='_left', rsuffix='_right'):
        """
        Args:
            estimator (BaseEstimator): Sklearn Estimator
            prefunc (callable):
            scoreplan(dict):
        """
        if estimator is not None:
            assert issubclass(type(estimator), BaseEstimator)
            self.estimator = estimator
        else:
            self.estimator = RandomForestClassifier(n_estimators=10)

        if prefunc is not None:
            assert isinstance(prefunc, callable)
        self.prefunc = prefunc
        assert isinstance(scoreplan, dict)
        self.scoreplan = scoreplan
        self._allcols = scoreplan.keys()
        self._id_cols = list()
        self._code_cols = list()
        self._text_cols = list()
        self._cat_cols = list()
        self._stop_words = dict()
        self._thresholds = dict()
        self._tokenizers = dict()
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self._ixnameleft = self.ixname + self.lsuffix
        self._ixnameright = self.ixname + self.rsuffix
        self._ixnamepairs = [self._ixnameleft, self._ixnameright]
        self._skmodel = BaseEstimator()
        self._vocab = dict()
        self._scorenames = dict()
        self._sbspipeplan = dict()
        for c in self._allcols:
            scoretype = self.scoreplan[c]['type']
            if scoretype == 'Id':
                self._id_cols.append(c)
                self._scorenames[c] = [c + '_exact']
            elif scoretype == 'Category':
                self._cat_cols.append(c)
                self._scorenames[c] = [c + '_exact']
                self._sbspipeplan[c] = ['exact']
            elif scoretype == 'Code':
                self._code_cols.append(c)
                self._scorenames[c] = [c + '_exact']
                self._sbspipeplan[c] = ['exact']
            elif scoretype == 'FreeText':
                self._text_cols.append(c)
                if self.scoreplan[c].get('stop_words') is not None:
                    self._stop_words[c] = self.scoreplan[c].get('stop_words')
                if self.scoreplan[c].get('threshold') is None:
                    self._thresholds[c] = _tfidf_threshold_value
                else:
                    assert self.scoreplan[c].get('threshold') <= 1.0
                    self._thresholds[c] = self.scoreplan[c].get('threshold')
                self._tokenizers[c] = oaacomparators.TfidfVectorizer(
                    stop_words=self._stop_words.get(c)
                )
                self._vocab[c] = list()
                self._scorenames[c] = [
                    c + '_exact',
                    c + '_token_wostopwords',
                    c + '_token',
                    c + '_tfidf'
                ]
                self._sbspipeplan[c] = ['exact', 'token']
                self._sbspipeplan[c + '_wostopwords'] = ['token']
        pass

    def _tfidf_transform(self, left, right, on, refit=True):
        """
        Args:
            left (pd.Series):
            right (pd.Series):
            on (str):
            refit (bool):
        Returns:
            pd.DataFrame
        """
        scorename = on + '_tfidf'
        left = left.dropna().copy()
        right = right.dropna().copy()
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        if refit is True:
            self._tfidf_fit(left=left, right=right, on=on, addvocab=refit)
        left_tfidf = self._tokenizers[on].transform(left)
        right_tfidf = self._tokenizers[on].transform(right)
        X = pd.DataFrame(
            cosine_similarity(left_tfidf, right_tfidf),
            columns=right.index
        )
        X[self._ixnameleft] = left.index
        score = pd.melt(
            X,
            id_vars=self._ixnameleft,
            var_name=self._ixnameright,
            value_name=scorename
        ).set_index(
            self._ixnamepairs
        )
        return score

    def _tfidf_fit(self, left, right, on, addvocab=True):
        """
        update the vocabulary of the tokenizer
        Args:
            left (pd.Series):
            right (pd.Series):
            on (str):
            addvocab (bool):

        Returns:
            self
        """
        left = left.dropna().copy().tolist()
        right = right.dropna().copy().tolist()
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        vocab2 = left + right
        if addvocab is True:
            assert isinstance(self._vocab[on], list)
            self._vocab[on] += vocab2
        else:
            self._vocab[on] = vocab2
        self._tokenizers[on].fit(self._vocab[on])
        return self

    def _id_transform(self, left, right, on):
        """

        Args:
            left (pd.Series):
            right (pd.Series):
            on (str):

        Returns:
            pd.DataFrame
        """
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        left = pd.DataFrame(left.dropna().copy()).reset_index(drop=False)
        right = pd.DataFrame(right.dropna().copy()).reset_index(drop=False)
        scorename = on + '_exact'
        x = pd.merge(left=left, right=right, left_on=on, right_on=on, how='inner',
                     suffixes=[self.lsuffix, self.rsuffix])
        x = x[self._ixnamepairs].set_index(self._ixnamepairs)
        x[scorename] = 1
        return x

    def _pruning_fit(self, left, right, addvocab=True):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            addvocab (bool):

        Returns:

        """
        for c in self._text_cols:
            self._tfidf_fit(left=left[c], right=right[c], on=c, addvocab=addvocab)
        return self

    def _pruning_row(self, row):
        """
        Checks that the row has either one id matching or is greater than the tfidf threshold defined
        Args:
            row (pd.Series):

        Returns:
            bool
        """
        for c in self._id_cols:
            if row[c] == 1:
                return True
        for c in self._text_cols:
            if row[c] is not None and row[c] >= self._thresholds[c]:
                return True
        return False

    def _pruning_transform(self, left, right):
        """

        Args:
            left (pd.DataFrame): {ix: [cols]}
            right (pd.DataFrame): {ix: [cols]}

        Returns:
            pd.DataFrame {[ix_left, ix_right]: [scores]}
        """
        scores = []
        for c in self._text_cols:
            tfidf_score = self._tfidf_transform(left=left[c], right=right[c], on=c, refit=True)
            scores.append(tfidf_score)
            del tfidf_score
        for c in self._id_cols:
            idscore = self._id_transform(left=left[c], right=right[c], on=c)
            scores.append(idscore)
        if len(scores) > 0:
            connectscores = oaacomparators.mergescore(scores)
            connectscores = connectscores[
                connectscores.apply(self._pruning_row, axis=1)
            ]
            connectscores.fillna(0, inplace=True)
        else:
            connectscores = connectors.cartesian_join(left[[]], right[[]])
        return connectscores

    def fit(self, left, right, pairs, addvocab=True):
        newleft = self.prefunc(left)
        newright = self.prefunc(right)
        assert isinstance(newleft, pd.DataFrame)
        assert isinstance(newright, pd.DataFrame)
        self._pruning_fit(left=newleft, right=newright, addvocab=addvocab)
        connectscores = self._pruning_transform(left=newleft, right=newright)
        ixtrain = pairs.index.intersection(connectscores.index)
        X_sbs = connectors.createsbs(pairs=connectscores, left=newleft, right=newright)
        X_train = X_sbs.loc[ixtrain]
        y_train = pairs.loc[ixtrain, 'y_true']
        dp_connectscores = sbscomparators.DataPasser(on_cols=connectscores.columns.tolist())
        sbspipe = sbscomparators.PipeSbsComparator(
            scoreplan=self._sbspipeplan
        )
        # noinspection PyTypeChecker
        scorer = make_union(
            *[
                sbspipe,
                dp_connectscores
            ]
        )
        ## Estimator
        imp = Imputer()
        self._skmodel = make_pipeline(
            *[
                scorer,
                imp,
                self.estimator
            ]
        )
        self._skmodel.fit(X_train, y_train)
        return self
