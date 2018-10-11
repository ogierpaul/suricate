import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, partial_token_set_ratio as tokenratio
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
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
            prefunc (callable): Add features to the data. Default lambda df: df
            scoreplan(dict):
        """
        if estimator is not None:
            assert issubclass(type(estimator), BaseEstimator)
            self.estimator = estimator
        else:
            self.estimator = RandomForestClassifier(n_estimators=100)

        if prefunc is not None:
            assert callable(prefunc)
        else:
            prefunc = lambda df: df
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
        self._suffixfuzzy = 'fuzzy'
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
        self.verbose = verbose
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
                self._vocab[actualcolname] = list()
                self._scorenames[actualcolname] = [
                    '_'.join([c, self._suffixwosw, self._suffixtoken]),
                    '_'.join([c, self._suffixtfidf]),
                    '_'.join([c, self._suffixtoken]),
                    '_'.join([c, self._suffixfuzzy])
                ]
                self._sbspipeplan[actualcolname] = [self._suffixtoken, self._suffixfuzzy]
                self._sbspipeplan[actualcolname + '_' + self._suffixwosw] = [self._suffixtoken]

                if self.scoreplan[c].get('stop_words') is not None:
                    self._stop_words[actualcolname] = self.scoreplan[c].get('stop_words')
                if self.scoreplan[c].get('threshold') is not None:
                    assert self.scoreplan[c].get('threshold') <= 1.0
                self._thresholds[actualcolname] = self.scoreplan[c].get('threshold')
                self._tokenizers[actualcolname] = self._tokenizermodel(
                    stop_words=self._stop_words.get(actualcolname)
                )

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

    def _tfidf_transform(self, left, right, on, addvocab='add', prune=False):
        """
        Args:
            left (pd.Series):
            right (pd.Series):
            on (str):
            addvocab (str): 'add', 'keep', 'replace'
        Returns:
            pd.Series
        """
        scorename = on + '_' + self._suffixtfidf
        left = left.dropna().copy()
        right = right.dropna().copy()
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        # If we cannot find a single value we return blank
        if left.shape[0] == 0 or right.shape[0] == 0:
            ix = pd.MultiIndex(levels=[[], []],
                               labels=[[], []],
                               names=self._ixnamepairs
                               )
            r = pd.Series(index=ix, name=scorename)
            return r
        if addvocab in ['add', 'replace']:
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
        if prune:
            ths = self._thresholds.get(on)
            if ths is None:
                ths = _tfidf_threshold_value
            score = score[score[scorename] > ths]
        score = score[scorename]
        return score

    def _tfidf_fit(self, left, right, on, addvocab='add'):
        """
        update the vocabulary of the tokenizer
        Args:
            left (pd.Series):
            right (pd.Series):
            on (str):
            addvocab (str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new

        Returns:
            self
        """
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        assert addvocab in ['add', 'keep', 'replace']
        if addvocab == 'keep':
            return self
        else:
            left = left.dropna().copy().tolist()
            right = right.dropna().copy().tolist()
            vocab2 = left + right
            if addvocab == 'add':
                assert isinstance(self._vocab[on], list)
                self._vocab[on] += vocab2
            elif addvocab == 'replace':
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

    def _pruning_fit(self, left, right, addvocab='add'):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            addvocab (str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new
        Returns:

        """
        for c in self._text_cols:
            self._tfidf_fit(left=left[c], right=right[c], on=c, addvocab=addvocab)
        return self

    def _pruning_transform(self, left, right, addvocab='add', verbose=False):
        """

        Args:
            left (pd.DataFrame): {ix: [cols]}
            right (pd.DataFrame): {ix: [cols]}
            addvocab (str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new

        Returns:
            pd.DataFrame {[ix_left, ix_right]: [scores]}
        """
        scores = dict()
        ix_taken = pd.MultiIndex(levels=[[], []],
                                 labels=[[], []],
                                 names=self._ixnamepairs
                                 )
        for c in self._id_cols:
            idscore = self._id_transform(left=left[c], right=right[c], on=c)
            ix_taken = ix_taken.union(
                idscore.index
            )
            scores[c] = idscore
            del idscore
        for c in self._text_cols:
            tfidf_score = self._tfidf_transform(left=left[c], right=right[c], on=c, addvocab=addvocab, prune=True)
            if self._thresholds.get(c) is not None:
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
        if verbose is True:
            possiblepairs = left.shape[0] * right.shape[0]
            actualpairs = connectscores.shape[0]
            compression = int(possiblepairs / actualpairs)
            print(
                '{} | Pruning compression factor of {} on {} possibles pairs'.format(
                    pd.datetime.now(), compression, possiblepairs
                )
            )
        return connectscores

    def _connectscores(self, left, right, addvocab='add', verbose=False):
        """

        Args:
            left: {ix: [cols]}
            right: {ix: [cols]}
            addvocab (str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new
        Returns:
            pd.DataFrame
        """
        X_connect = self._pruning_transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
        X_sbs = connectors.createsbs(pairs=X_connect, left=left, right=right)
        return X_sbs

    def fit(self, left, right, pairs, addvocab='add', verbose=None):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            pairs (pd.Series/pd.DataFrame):
            addvocab(str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new

        Returns:
            self
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print('{} | Start fit'.format(pd.datetime.now()))
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        if isinstance(pairs, pd.DataFrame):
            pairs = pairs['y_true']
        y_train = pairs

        self._pruning_fit(left=newleft, right=newright, addvocab=addvocab)
        X_sbs = self._connectscores(left=newleft, right=newright, addvocab=addvocab, verbose=True)
        if X_sbs.shape[0] == 0:
            raise Warning(
                'No possible matches based on current pruning --> not possible to fit.',
                '\n - Provide better training data',
                '\n - Check the pruning threshold and ids',
            )

        if verbose:
            precision, recall = _evalprecisionrecall(y_true=y_train, y_catched=X_sbs)
            print('{} | Pruning score: precision: {:.2%}, recall: {:.2%}'.format(pd.datetime.now(), precision, recall))
        # Expand X_sbs to:
        # - include positives samples not included in pruning
        # - fill those rows with 0 for the pruning scores
        true_pos = y_train.loc[y_train == 1]
        missed_pos = true_pos.loc[
            true_pos.index.difference(
                X_sbs.index
            )
        ]
        X_missed = pd.DataFrame(
            index=missed_pos,
            columns=X_sbs.columns
        )
        X_missed[self._connectcols] = 0
        X_sbs = pd.concat(
            [X_sbs, X_missed],
            axis=0, ignore_index=False
        )
        # Redefine y_train to have a common index with x_sbs
        y_train = y_train.loc[
            y_train.index.intersection(
                X_sbs.index
            )
        ]
        X_train = X_sbs.loc[
            y_train.index
        ]
        self._skmodel.fit(X_train, y_train)
        if verbose:
            scores = self.score(left=left, right=right, pairs=pairs, kind='all')
            assert isinstance(scores, dict)
            print(
                '{} | Model score: precision: {:.2%}, recall: {:.2%}'.format(
                    pd.datetime.now(),
                    scores['precision'],
                    scores['recall']
                )
            )
            pass
        return self

    def _pruning_pred(self, left, right, pairs, addvocab='add'):
        """

        Args:
            left:
            right:
            pairs (pd.Series/pd.DataFrame):
            addvocab:

        Returns:

        """
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        if isinstance(pairs, pd.DataFrame):
            pairs = pairs['y_true']
        y_true = pairs
        X_sbs = self._connectscores(left=newleft, right=newright, addvocab=addvocab)
        y_pred = pd.Series(index=X_sbs.index).fillna(1)
        y_pred = connectors.indexwithytrue(y_true=y_true, y_pred=y_pred)
        return y_pred

    def predict_proba(self, left, right, addvocab='add', verbose=False, addmissingleft=False) -> pd.Series:
        """

        Args:
            left:
            right:
            addvocab:
            verbose:

        Returns:
            pd.Series
        """

        if verbose:
            print('{} | Start pred'.format(pd.datetime.now()))
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        X_sbs = self._connectscores(left=newleft, right=newright, addvocab=addvocab)

        # if we have results
        if X_sbs.shape[0] > 0:
            df_pred = self._skmodel.predict_proba(X_sbs)
            y_proba = pd.DataFrame(
                df_pred,
                index=X_sbs.index
            )[1]
            y_proba.name = 'y_proba'
        else:
            # Create an empty recipient
            y_proba = pd.Series(
                index=pd.MultiIndex(
                    levels=[[], []],
                    labels=[[], []],
                    names=self._ixnamepairs
                ),
                name='y_proba'
            )
        # add a case for missing lefts
        if addmissingleft is True:
            missing_lefts = newleft.index.difference(
                X_sbs.index.get_level_values(self._ixnameleft)
            )
            missing_lefts = pd.MultiIndex.from_product(
                [
                    missing_lefts,
                    [None]
                ],
                names=self._ixnamepairs
            )
            missing_lefts = pd.Series(
                index=missing_lefts,
                name='y_proba'
            ).fillna(
                0
            )
            y_proba = pd.concat([y_proba, missing_lefts], axis=0)
        assert isinstance(y_proba, pd.Series)
        assert y_proba.name == 'y_proba'
        assert y_proba.index.names == self._ixnamepairs
        return y_proba

    def predict(self, left, right, addvocab='add', verbose=False, addmissingleft=False):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            addvocab(str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new

        Returns:
            pd.Series
        """
        y_proba = self.predict_proba(left=left, right=right, addvocab=addvocab,
                                     verbose=verbose, addmissingleft=addmissingleft)

        # noinspection PyUnresolvedReferences
        y_pred = (y_proba > 0.5).astype(float)
        assert isinstance(y_pred, pd.Series)
        y_pred.name = 'y_proba'
        return y_pred

    def _evalpruning(self, left, right, pairs, addvocab='add', verbose=False):
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        if isinstance(pairs, pd.DataFrame):
            pairs = pairs['y_true']
        y_train = pairs

        self._pruning_fit(left=newleft, right=newright, addvocab=addvocab)
        X_sbs = self._connectscores(left=newleft, right=newright, addvocab=addvocab, verbose=verbose)
        precision, recall = _evalprecisionrecall(y_true=y_train, y_catched=X_sbs)
        return precision, recall

    def score(self, left, right, pairs, kind='accuracy'):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            pairs (pd.DataFrame/ pd.Series):
            kind (str): 'accuracy', 'precision', 'recall', 'f1', 'all'
        Returns:
            float
        """
        assert kind in ['accuracy', 'precision', 'recall', 'f1', 'all']
        scores = self._scores(left=left, right=right, pairs=pairs)
        if kind != 'all':
            return scores[kind]
        else:
            return scores

    def _scores(self, left, right, pairs):
        """

        Args:
            left:
            right:
            pairs:

        Returns:
            dict
        """
        y_pred = self.predict(left=left, right=right)
        assert isinstance(y_pred, pd.Series)
        if isinstance(pairs, pd.DataFrame):
            pairs = pairs['y_true']
        y_true = pairs
        assert isinstance(y_true, pd.Series)
        scores = _metrics(y_true=y_true, y_pred=y_pred)
        return scores


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


def _evalprecisionrecall(y_true, y_catched):
    """

    Args:
        y_catched (pd.DataFrame/pd.Series):
        y_true (pd.Series):

    Returns:
        float, float: precision and recall
    """
    true_pos = y_true.loc[y_true == 1]
    true_neg = y_true.loc[y_true == 0]
    catched_pos = y_catched.loc[true_pos.index.intersection(y_catched.index)]
    catched_neg = y_catched.loc[y_catched.index.difference(catched_pos.index)]
    missed_pos = true_pos.loc[true_pos.index.difference(y_catched.index)]
    assert true_pos.shape[0] + true_neg.shape[0] == y_true.shape[0]
    assert catched_pos.shape[0] + catched_neg.shape[0] == y_catched.shape[0]
    assert catched_pos.shape[0] + missed_pos.shape[0] == true_pos.shape[0]
    recall = catched_pos.shape[0] / true_pos.shape[0]
    precision = catched_pos.shape[0] / y_catched.shape[0]
    return precision, recall


def _metrics(y_true, y_pred):
    y_pred2 = connectors.indexwithytrue(y_true=y_true, y_pred=y_pred)
    scores = dict()
    scores['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred2)
    scores['precision'] = precision_score(y_true=y_true, y_pred=y_pred2)
    scores['recall'] = recall_score(y_true=y_true, y_pred=y_pred2)
    scores['f1'] = f1_score(y_true=y_true, y_pred=y_pred2)
    return scores


def _evalpred(y_true, y_pred, verbose=True, set=None):
    if set is None:
        sset = ''
    else:
        sset = 'for set {}'.format(set)
    precision, recall = _evalprecisionrecall(y_true=y_true, y_catched=y_pred)
    if verbose:
        print(
            '{} | Pruning score: precision: {:.2%}, recall: {:.2%} {}'.format(
                pd.datetime.now(), precision, recall, sset
            )
        )
    scores = _metrics(y_true=y_true, y_pred=y_pred)
    if verbose:
        print(
            '{} | Model score: precision: {:.2%}, recall: {:.2%} {}'.format(
                pd.datetime.now(), scores['precision'], scores['recall'], sset
            )
        )
    return scores
