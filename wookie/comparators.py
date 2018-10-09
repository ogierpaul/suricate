import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, partial_token_set_ratio as tokenratio
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

import wookie.preutils
from wookie import connectors

_navalue_score = None

_tfidf_threshold_value = 0.3


class BaseSbsComparator(TransformerMixin):
    def __init__(self, left='left', right='right', compfunc=None, *args, **kwargs):
        """
        base class for all transformers
        Args:
            left (str):
            right (str):
            compfunc (function): ['fuzzy', 'token', 'exact']
        """
        TransformerMixin.__init__(self)
        self.left = left
        self.right = right
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc

    def transform(self, X):
        """
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        compfunc = self.compfunc
        if not compfunc is None:
            y = X.apply(
                lambda r: compfunc(
                    r.loc[self.left],
                    r.loc[self.right]
                ),
                axis=1
            ).values.reshape(-1, 1)
            return y
        else:
            raise ValueError('compfunc is not defined')

    def fit(self, *_):
        return self


class DataPasser(TransformerMixin):
    """
    This dont do anything, just pass the data on selected columns
    if on_cols is None, pass the whole dataframe
    """

    def __init__(self, on_cols=None):
        TransformerMixin.__init__(self)
        self.on_cols = on_cols

    def fit(self, X=None, y=None):

        return self

    def transform(self, X):
        """
        This dont do anything, just pass the data as it is
        Args:
            X:

        Returns:

        """
        if not self.on_cols is None:
            assert isinstance(X, pd.DataFrame)
            res = X[self.on_cols]
        else:
            res = X
        return res


class FuzzyWuzzySbsComparator(BaseSbsComparator, TransformerMixin):
    """
    Compare two columns of a dataframe with one another using functions from fuzzywuzzy library
    """

    def __init__(self, comparator=None, left='left', right='right', *args, **kwargs):
        """
        Args:
            comparator (str): name of the comparator function: ['exact', 'fuzzy', 'token']
            left (str): name of left column
            right (str): name of right column
            *args:
            **kwargs:
        """
        if comparator == 'exact':
            compfunc = _exact_score
        elif comparator == 'fuzzy':
            compfunc = _fuzzy_score
        elif comparator == 'token':
            compfunc = _token_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token']")
        BaseSbsComparator.__init__(
            self,
            compfunc=compfunc,
            left=left,
            right=right,
            *args,
            **kwargs
        )
        pass


class PipeSbsComparator(TransformerMixin):
    """
    Align several FuzzyWuzzyComparator
    Provided that the column are named:
    comp1 = PipeComparator(
    scoreplan={
        'name': ['exact', 'fuzzy', 'token'],
        'street': ['exact', 'token'],
        'duns': ['exact'],
        'city': ['fuzzy'],
        'postalcode': ['exact'],
        'country_code':['exact']
    }
)
    """

    def __init__(self, scoreplan):
        """

        Args:
            scoreplan (dict): of type {'col': 'comparator'}
        """
        TransformerMixin.__init__(self)
        self.scoreplan = scoreplan

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, n_jobs=1, *args, **kwargs):
        """

        Args:
            X (pd.DataFrame):
            n_jobs (int): number of jobs
            *args:
            **kwargs:

        Returns:
            np.ndarray
        """
        stages = []
        for k in self.scoreplan.keys():
            left = '_'.join([k, 'left'])
            right = '_'.join([k, 'right'])
            for v in self.scoreplan[k]:
                stages.append(
                    FuzzyWuzzySbsComparator(left=left, right=right, comparator=v)
                )
        pipe = make_union(n_jobs=n_jobs, *stages, *args, **kwargs)
        res = pipe.fit_transform(X)

        return res

class LrDuplicateFinder:
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

    def __init__(self, scoreplan, prefunc=None, estimator=None, ixname='ix', lsuffix='_left', rsuffix='_right',
                 verbose=False):
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
            assert callable(prefunc)
        self.prefunc = prefunc
        assert isinstance(scoreplan, dict)
        self.scoreplan = scoreplan
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        # Derive the new column names
        self._ixnameleft = self.ixname + self.lsuffix
        self._ixnameright = self.ixname + self.rsuffix
        self._ixnamepairs = [self._ixnameleft, self._ixnameright]
        self._suffixascii = 'ascii'
        self._suffixwosw = 'wostopwords'
        self._suffixid = 'cleaned'
        self._suffixexact = 'exact'
        self._suffixtoken = 'token'
        self._suffixtfidf = 'tfidf'
        # From score plan initiate the list of columns
        self._usedcols = scoreplan.keys()
        self._id_cols = list()
        self._code_cols = list()
        self._text_cols = list()
        self._cat_cols = list()
        # Initiate the list of values used for the FreeTextAnalyszs
        self._stop_words = dict()
        self._thresholds = dict()
        self._tokenizers = dict()
        self._vocab = dict()
        self._scorenames = dict()
        self._sbspipeplan = dict()
        self._tokenizermodel = TfidfVectorizer
        self.verbose = True
        # Initiate for subclass the scikit-learn pipeline
        self._skmodel = BaseEstimator()
        # Loop through the score plan
        for c in self._usedcols:
            scoretype = self.scoreplan[c]['type']
            if scoretype in ['Id', 'Category', 'Code']:
                actualcolname = '_'.join([c, self._suffixid])
                self._scorenames[c] = ['_'.join([actualcolname, self._suffixexact])]
                if scoretype == 'Id':
                    self._id_cols.append(actualcolname)
                else:
                    self._sbspipeplan[actualcolname] = [self._suffixexact]
                    if scoretype == 'Category':
                        self._cat_cols.append(actualcolname)
                    elif scoretype == 'Code':
                        self._code_cols.append(actualcolname)
            elif scoretype == 'FreeText':
                actualcolname = '_'.join([c, self._suffixascii])
                self._text_cols.append(actualcolname)
                if self.scoreplan[c].get('stop_words') is not None:
                    self._stop_words[actualcolname] = self.scoreplan[c].get('stop_words')
                if self.scoreplan[c].get('threshold') is None:
                    self._thresholds[actualcolname] = _tfidf_threshold_value
                else:
                    assert self.scoreplan[c].get('threshold') <= 1.0
                    self._thresholds[actualcolname] = self.scoreplan[c].get('threshold')
                self._tokenizers[actualcolname] = self._tokenizermodel(
                    stop_words=self._stop_words.get(actualcolname)
                )
                self._vocab[actualcolname] = list()
                self._scorenames[actualcolname] = [
                    c + '_' + self._suffixexact,
                    c + '_' + self._suffixwosw + '_' + self._suffixtoken,
                    c + '_' + self._suffixtoken,
                    c + '_' + self._suffixtfidf
                ]
                self._sbspipeplan[actualcolname] = [self._suffixexact, self._suffixtoken]
                self._sbspipeplan[actualcolname + '_' + self._suffixwosw] = [self._suffixtoken]
        self._connectcols = [
                                c + '_' + self._suffixtfidf for c in self._text_cols
                            ] + [
                                c + '_' + self._suffixexact for c in self._id_cols
                            ]
        dp_connectscores = wookie.comparators.DataPasser(on_cols=self._connectcols)
        sbspipe = wookie.comparators.PipeSbsComparator(
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
        pass

    def _tfidf_transform(self, left, right, on, addvocab=True, prune=False):
        """
        Args:
            left (pd.Series):
            right (pd.Series):
            on (str):
            addvocab (bool):
        Returns:
            pd.Series
        """
        scorename = on + '_' + self._suffixtfidf
        left = left.dropna().copy()
        right = right.dropna().copy()
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        if addvocab is True:
            self._tfidf_fit(left=left, right=right, on=on, addvocab=addvocab)
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
        if prune is True:
            ths = self._thresholds[on]
            if ths is None:
                ths = _tfidf_threshold_value
            score = score[score[scorename] > ths]
        score = score[scorename]
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
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        left = left.dropna().copy().tolist()
        right = right.dropna().copy().tolist()
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
            pd.Series
        """
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        left = pd.DataFrame(left.dropna().copy()).reset_index(drop=False)
        right = pd.DataFrame(right.dropna().copy()).reset_index(drop=False)
        scorename = on + '_' + self._suffixexact
        x = pd.merge(left=left, right=right, left_on=on, right_on=on, how='inner',
                     suffixes=[self.lsuffix, self.rsuffix])
        x = x[self._ixnamepairs].set_index(self._ixnamepairs)
        x[scorename] = 1
        x = x[scorename]
        return x

    def _prepare_data(self, df):
        """

        Args:
            df (pd.DataFrame):

        Returns:
            new (pd.DataFrame)
        """
        new = df.copy()
        for c in self.scoreplan.keys():
            scoretype = self.scoreplan[c]['type']
            if scoretype == 'FreeText':
                actualcolname = '_'.join([c, self._suffixascii])
                new[actualcolname] = new[c].apply(wookie.preutils.lowerascii)
                if actualcolname in self._stop_words.keys():
                    colnamewosw = '_'.join([actualcolname, self._suffixwosw])
                    new[colnamewosw] = new[actualcolname].apply(
                        lambda r: wookie.preutils.rmvstopwords(r, stop_words=self._stop_words[actualcolname])
                    )
            else:
                actualcolname = '_'.join([c, self._suffixid])
                new[actualcolname] = new[c].apply(
                    wookie.preutils.lowerascii
                ).apply(
                    wookie.preutils.idtostr
                )
        assert isinstance(new, pd.DataFrame)
        return new

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

    def _pruning_transform(self, left, right, addvocab=True):
        """

        Args:
            left (pd.DataFrame): {ix: [cols]}
            right (pd.DataFrame): {ix: [cols]}
            addvocab (bool)

        Returns:
            pd.DataFrame {[ix_left, ix_right]: [scores]}
        """
        scores = dict()
        ix_taken = pd.MultiIndex(levels=[[], []],
                                 labels=[[], []],
                                 names=self._ixnamepairs)
        for c in self._id_cols:
            idscore = self._id_transform(left=left[c], right=right[c], on=c)
            ix_taken = ix_taken.union(
                idscore.index
            )
            scores[c] = idscore
            del idscore
        for c in self._text_cols:
            tfidf_score = self._tfidf_transform(left=left[c], right=right[c], on=c, addvocab=addvocab, prune=True)
            if self._thresholds[c] is not None:
                ix_taken = ix_taken.union(
                    tfidf_score.loc[tfidf_score > self._thresholds[c]].index
                )
            scores[c] = tfidf_score
            del tfidf_score
        connectscores = pd.DataFrame(index=ix_taken)
        for c in self._text_cols:
            connectscores[c + '_' + self._suffixtfidf] = scores[c].loc[ix_taken.intersection(scores[c].index)]
        for c in self._id_cols:
            connectscores[c + '_' + self._suffixexact] = scores[c].loc[ix_taken.intersection(scores[c].index)]
        # case we have no id cols or text cols make a cartesian joi
        if connectscores.shape[1] == 0:
            connectscores = connectors.cartesian_join(left[[]], right[[]])
        if self.verbose is True:
            possiblepairs = left.shape[0] * right.shape[0]
            actualpairs = connectscores.shape[0]
            compression = int(possiblepairs / actualpairs)
            print('Compression of {} : {} of {} selected'.format(compression, actualpairs, possiblepairs))
        return connectscores

    def fit(self, left, right, pairs, addvocab=True):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            pairs (pd.DataFrame):
            addvocab(bool):

        Returns:
            self
        """
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        assert isinstance(newleft, pd.DataFrame)
        assert isinstance(newright, pd.DataFrame)
        self._pruning_fit(left=newleft, right=newright, addvocab=addvocab)
        connectscores = self._pruning_transform(left=newleft, right=newright)
        ixtrain = pairs.index.intersection(connectscores.index)
        X_sbs = connectors.createsbs(pairs=connectscores, left=newleft, right=newright)
        X_train = X_sbs.loc[ixtrain]
        y_train = pairs.loc[ixtrain, 'y_true']
        self._skmodel.fit(X_train, y_train)
        return self

    def predict(self, left, right, addvocab=True):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            addvocab(bool):

        Returns:
            pd.Series
        """
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        assert isinstance(newleft, pd.DataFrame)
        assert isinstance(newright, pd.DataFrame)
        connectscores = self._pruning_transform(left=newleft, right=newright)
        X_sbs = connectors.createsbs(pairs=connectscores, left=newleft, right=newright)
        y_pred = self._skmodel.predict(X_sbs)
        y_pred = pd.Series(y_pred, index=X_sbs.index)
        return y_pred

    def score(self, left, right, pairs):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            pairs (pd.DataFrame):

        Returns:
            float
        """
        y_pred = self.predict(left=left, right=right)
        assert isinstance(y_pred, pd.Series)
        y_true = pairs['y_true']
        assert isinstance(y_true, pd.Series)
        y_pred2 = pd.Series(index=y_true.index)
        y_pred2.loc[y_true.index.intersection(y_pred.index)] = y_pred
        y_pred2.fillna(0, inplace=True)
        score = accuracy_score(y_true=y_true, y_pred=y_pred2)
        return score


def _preparevocab(on, left, right):
    """
    create a series with all the vocab wors
    Args:
        on (str): column to take for the vocabulary
        left (pd.DataFrame):
        right (pd.DataFrame)
    Returns:
        pd.Series
    """
    vocab = pd.Series(name=on)
    for df in [left, right]:
        vocab = pd.concat([vocab, df[on].dropna()], axis=0, ignore_index=True)
    return vocab


def innermatch(left, right, on, ixname='ix', lsuffix='_left', rsuffix='_right'):
    """
    Gives the indexes of the two dataframe where the ids are matching
    Args:
        left (pd.DataFrame): left df of the form {ixname:['name',..]}
        right (pd.DataFrame): right df of the form {ixname:['name',..]}
        on (str): name of the column
        ixname (str): default 'ix'
        lsuffix (str): default '_left'
        rsuffix (str): default '_right'
    Returns:
        {['ix_left', 'ix_right']: [id_exact]}
    """

    left = left[[on]].dropna().copy().reset_index(drop=False)
    right = right[[on]].dropna().copy().reset_index(drop=False)
    ix_left = ixname + lsuffix
    ix_right = ixname + rsuffix
    scorename = on + '_exact'
    x = pd.merge(left=left, right=right, left_on=on, right_on=on, how='inner', suffixes=[lsuffix, rsuffix])
    x = x[[ix_left, ix_right]].set_index([ix_left, ix_right])
    x[scorename] = 1
    return x


def _valid_inputs(left, right):
    """
    takes two inputs and return True if none of them is null, or False otherwise
    Args:
        left: first object (scalar)
        right: second object (scalar)

    Returns:
        bool
    """
    if any(pd.isnull([left, right])):
        return False
    else:
        return True


def _exact_score(left, right):
    """
    Checks if the two values are equali
    Args:
        left (object): object number 1
        right (object): object number 2

    Returns:
        float
    """
    if _valid_inputs(left, right) is False:
        return _navalue_score
    else:
        return float(left == right)


def _fuzzy_score(left, right):
    """
    return ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if _valid_inputs(left, right) is False:
        return _navalue_score
    else:
        s = (simpleratio(left, right) / 100)
    return s


def _token_score(left, right):
    """
    return the token_set_ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if _valid_inputs(left, right) is False:
        return _navalue_score
    else:
        s = tokenratio(left, right) / 100
    return s
