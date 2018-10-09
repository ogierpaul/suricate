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


def _tokencompare(left, right, tokenizer, ix_left='ix_left', ix_right='ix_right', fit=False):
    """
    return the cosine similarity of the tf-idf score of each possible pairs of documents
    Args:
        left (pd.Series): new data (To compare against train data)
        right (pd.Series): train data (To fit the tf-idf transformer)
        tokenizer: tokenizer of type sklearn
        ix_left (str): column name of new ix
        ix_right (str): column name of train ix
        fit (bool): if False, do not fit the tokenizer (is already fitted)
    Returns:
        pd.DataFrame
    """
    right = right.copy().dropna()
    left = left.copy().dropna()
    if fit is True:
        alldata = pd.concat([right, left], axis=0, ignore_index=True)
        tokenizer.fit(alldata)
    right_tfidf = tokenizer.predict(right.dropna())
    left_tfidf = tokenizer.predict(left)
    X = pd.DataFrame(cosine_similarity(left_tfidf, right_tfidf), columns=right.index)
    X[ix_left] = left.index
    score = pd.melt(
        X,
        id_vars=ix_left,
        var_name=ix_right,
        value_name='score'
    ).sort_values(by='score', ascending=False)
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


def idsconnector(left, right, on_cols, ixname='ix', lsuffix='_left', rsuffix='_right'):
    """
        Gives the indexes of the two dataframe where the ids are matching
    Args:
        left (pd.DataFrame): left df of the form {ixname:['name',..]}
        right (pd.DataFrame): right df of the form {ixname:['name',..]}
        on_cols (list): names of the columns
        ixname (str): default 'ix'
        lsuffix (str): default '_left'
        rsuffix (str): default '_right'
    Returns:
        {['ix_left', 'ix_right']: [id_exact, id2_exact]}
    """
    alldata = None
    for c in on_cols:
        pos = innermatch(left=left, right=right, on=c, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        if alldata is None:
            alldata = pos.copy()
        else:
            alldata = pd.merge(left=alldata, right=pos, left_index=True, right_index=True, how='outer')
    alldata.fillna(0, inplace=True)
    return alldata


def tfidfconnector(left, right, on, threshold=0, ixname='ix', lsuffix='_left', rsuffix='_right', **kwargs):
    """

    Args:
        left (pd.DataFrame): left df of the form {ixname:['name',..]}
        right (pd.DataFrame): right df of the form {ixname:['name',..]}
        on (str): column on which to calculate the tfidf similarity score
        ixname (str): default 'ix'
        lsuffix (str): default '_left'
        rsuffix (str): default '_right'
        threshold: Threshold on which to filter the pairs
        **kwargs: kwargs for the scikit learn TfIdf Vectorizer

    Returns:
        pd.DataFrame {[ixname+lsuffix, ixname+rsuffix]: [on+'_tfidf']}
    """
    left = left[on].dropna().copy()
    right = right[on].dropna().copy()
    assert isinstance(left, pd.Series)
    assert isinstance(right, pd.Series)
    ix_left = ixname + lsuffix
    ix_right = ixname + rsuffix
    scorename = on + '_tfidf'
    tokenizer = TfidfVectorizer(**kwargs)
    vocab = pd.concat(
        [left, right],
        axis=0,
        ignore_index=True
    )
    tokenizer.fit(vocab)
    right_tfidf = tokenizer.transform(right.dropna())
    left_tfidf = tokenizer.transform(left)
    X = pd.DataFrame(cosine_similarity(left_tfidf, right_tfidf), columns=right.index)
    X[ix_left] = left.index
    score = pd.melt(
        X,
        id_vars=ix_left,
        var_name=ix_right,
        value_name=scorename
    ).set_index(
        [
            ix_left,
            ix_right
        ]
    ).sort_values(
        by=scorename,
        ascending=False
    )
    if not threshold is None:
        score = score.loc[score[scorename] > threshold]
    return score


def mergescore(scores):
    """
    Merge the scores of the tfidfconnector and of the idsconnector
    Args:
        scores (list): list of pd.DataFrame of the form {['ix_left', 'ix_right']: [score]}

    Returns:
        pd.DataFrame : of the form {['ix_left', 'ix_right']: [score1, score2, ...]}
    """
    X = None
    for s in scores:
        if X is None:
            X = s
        else:
            X = pd.merge(left=X, right=s, how='outer', left_index=True, right_index=True)
    return X


class LrTfidfConnector:
    def __init__(self, on, ixname='ix', lsuffix='_left', rsuffix='_right', threshold=0, **kwargs):
        self.on = on
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.vocab = pd.Series()
        self.ixnameleft = ixname + lsuffix
        self.ixnameright = ixname + rsuffix
        self.ixnamepairs = [self.ixnameleft, self.ixnameright]
        self.scorename = self.on + '_tfidf'
        self.threshold = threshold
        self.tokenizer = TfidfVectorizer(**kwargs)
        pass

    def fit(self, left, right, refit=True):
        """

        Args:
            left (pd.Series): {ixname: val}
            right (pd.Series): {ixname: val}
            refit (bool): whether to re-fit the data to new vocabulary

        Returns:

        """
        left = left.dropna().copy()
        right = right.dropna().copy()
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        vocab2 = pd.concat(
            [left, right],
            axis=0,
            ignore_index=True
        )
        if refit is True:
            self.vocab = pd.concat(
                [self.vocab, vocab2],
                axis=0,
                ignore_index=True
            )
        else:
            self.vocab = vocab2
        self.tokenizer.fit(self.vocab)
        return self

    def transform(self, left, right, refit=True):
        if refit is True:
            self.fit(left=left, right=right, refit=refit)
        left = left.dropna().copy()
        right = right.dropna().copy()
        left_tfidf = self.tokenizer.transform(left)
        right_tfidf = self.tokenizer.transform(right)
        X = pd.DataFrame(
            cosine_similarity(left_tfidf, right_tfidf),
            columns=right.index
        )
        X[self.ixnameleft] = left.index
        score = pd.melt(
            X,
            id_vars=self.ixnameleft,
            var_name=self.ixnameright,
            value_name=self.scorename
        ).set_index(
            self.ixnamepairs
        ).sort_values(
            by=self.scorename,
            ascending=False
        )
        if not self.threshold is None:
            score = score.loc[score[self.scorename] > self.threshold]
        return score


class IdTfIdfConnector:
    def __init__(self, tfidf_cols=None, stop_words=None, id_cols=None, threshold=0.3):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            tfidf_cols (list): ['name', 'street']
            stop_words (dict): {'name': ['inc', 'ltd'], 'street':['road', 'avenue']}
            id_cols (list): ['duns', 'euvat']
            threshold (float): 0.3
        """
        self.tfidf_cols = tfidf_cols
        self.stop_words = stop_words
        self.idcols = id_cols
        self.threshold = threshold
        self.tokenizers = dict()
        if not stop_words is None:
            assert isinstance(stop_words, dict)
        # Initiate the tokenizers
        if tfidf_cols is not None:
            assert hasattr(tfidf_cols, '__iter__')
            for c in tfidf_cols:
                if not stop_words is None and not stop_words.get(c) is None:
                    assert hasattr(stop_words[c], '__iter__')
                self.tokenizers[c] = self._init_tokenizer(on=c, stop_words=stop_words, threshold=threshold)
        if id_cols is not None:
            assert hasattr(id_cols, '__iter__')

    def _init_tokenizer(self, on, stop_words, threshold=0):
        """
        Initiate the Left-Right Tokenizer
        Args:
            on (str): column name
            stop_words (dict): dictionnnary of stopwords

        Returns:
            LrTfidfConnector
        """
        if stop_words is None:
            sw = None
        else:
            sw = stop_words.get(on)

        return LrTfidfConnector(on=on, stop_words=sw, threshold=threshold)

    def fit(self, left, right):
        for c in self.tfidf_cols:
            self.tokenizers[c].fit(left, right)
        pass

    def transform(self, left, right):
        scores = list()
        if self.tfidf_cols is not None:
            for c in self.tfidf_cols:
                tk = self.tokenizers[c]
                assert isinstance(tk, LrTfidfConnector)
                tfscore = tk.transform(left[c], right[c])
                scores.append(tfscore)
        if self.idcols is not None:
            idsscore = idsconnector(left=left, right=right, on_cols=self.idcols)
            scores.append(idsscore)
        if len(scores) > 0:
            connectscores = mergescore(scores)
        else:
            connectscores = connectors.cartesian_join(left[[]], right[[]])
        return connectscores


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


class _SbsTokenComparator(TransformerMixin):
    def __init__(self, tokenizer, ixnameleft='ix_left', ixnameright='ix_right', new_col='name', train_col='name'):
        """

        Args:
            tokenizer (TfidfVectorizer): Tokenizer
        """
        TransformerMixin.__init__(self)
        self.tokenizer = tokenizer
        self.new_ix = ixnameleft
        self.train_ix = ixnameright
        self.new_col = new_col
        self.train_col = train_col

    def fit(self, X=None, y=None):
        """
        Do Nothing
        Args:
            X: iterable
            y

        Returns:

        """
        return self

    def transform(self, X):
        """

        Args:
            X (pd.DataFrame):

        Returns:
            pd.DataFrame
        """
        ## format
        new_series = _prepare_deduped_series(X, ix=self.new_ix, val=self.new_col)
        train_series = _prepare_deduped_series(X, ix=self.train_ix, val=self.train_col)

        score = _tokencompare(
            right=train_series,
            left=new_series,
            tokenizer=self.tokenizer,
            ix_left=self.new_ix,
            ix_right=self.train_ix
        )
        score.set_index(
            [self.new_ix, self.train_ix],
            drop=True,
            inplace=True
        )
        score = score.loc[
            X.set_index(
                [self.new_ix, self.train_ix]
            ).index
        ]

        return score


def _prepare_deduped_series(X, ix, val):
    """
    deduplicate the records for one column based on one index column
    Args:
        X (pd.DataFrame)
        ix (str): name of index col
        val (str): name of value col
    Returns:
        pd.Series
    """
    y = X[
        [ix, val]
    ].drop_duplicates(
        subset=[ix]
    ).rename(
        columns={val: 'data'}
    ).set_index(
        ix, drop=True
    ).dropna(
        subset=['data']
    )['data']
    return y


_navalue_score = None


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


_tfidf_threshold_value = 0.3


class SbsModel:
    def __init__(self, prefunc, idtfidf, sbscomparator, estimator):
        self.prefunc = prefunc
        assert isinstance(idtfidf, IdTfIdfConnector)
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
        dp_connectscores = wookie.comparators.DataPasser(on_cols=connectscores.columns.tolist())
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
