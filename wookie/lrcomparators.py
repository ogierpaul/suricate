import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

# TO CHECK
import wookie.comparators
import wookie.preutils
from wookie import connectors
# noinspection PyProtectedMember
from wookie.comparators import _evalprecisionrecall, _metrics

# TODO: initiate X_sbs when Pipeplan is None

_tfidf__store_threshold_value = 0.5


class BaseLrComparator(TransformerMixin):
    def __init__(self,
                 on=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 scoresuffix='score',
                 store_threshold=0
                 ):
        TransformerMixin.__init__(self)
        self.on = on
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.scoresuffix = scoresuffix
        self.ixnameleft = '_'.join([self.ixname, self.lsuffix])
        self.ixnameright = '_'.join([self.ixname, self.rsuffix])
        self.ixnamepairs = [self.ixnameleft, self.ixnameright]
        self.store_threshold = store_threshold
        self.outcol = '_'.join([self.on, self.scoresuffix])
        pass

    def _dummyfit(self, left=None, right=None):
        # DO NOTHING
        return self

    def _dummytransform(self, left=None, right=None):
        newleft = left[[]]
        newright = right[[]]
        cart = connectors.cartesian_join(
            left_df=left,
            right_df=right,
            left_suffix=self.lsuffix,
            right_suffix=self.rsuffix
        )
        return cart

    def _toseries(self, left, right):
        """
        convert to series withoutnulls and copy
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):

        Returns:
            pd.Series, pd.Series
        """
        newleft = pd.Series()
        newright = pd.Series()
        if isinstance(left, pd.DataFrame):
            newleft = left[self.on].dropna().copy()
        if isinstance(right, pd.DataFrame):
            newright = right[self.on].dropna().copy()
        if isinstance(left, pd.Series):
            newleft = left.dropna().copy()
        if isinstance(right, pd.Series):
            newright = right.dropna().copy()
        for s, c in zip(['left', 'right'], [left, right]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        return newleft, newright

    def _todf(self, left, right):
        """
        convert to series withoutnulls and copy
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):

        Returns:
            pd.DataFrame, pd.DataFrame
        """
        newleft = pd.DataFrame()
        newright = pd.DataFrame()
        if isinstance(left, pd.DataFrame):
            newleft = left[[self.on]].dropna(subset=[self.on]).copy()
        if isinstance(right, pd.DataFrame):
            newright = right[[self.on]].dropna(subset=[self.on]).copy()
        if isinstance(left, pd.Series):
            newleft = pd.DataFrame(left.dropna().copy())
        if isinstance(right, pd.Series):
            newright = pd.DataFrame(right.dropna().copy())
        for s, c in zip(['left', 'right'], [left, right]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        return newleft, newright

    def _evalscore(self, left, right, y_true):
        # assert hasattr(self, 'transform') and callable(getattr(self, 'transform'))
        # noinspection
        y_pred = self.transform(left=left, right=right)
        precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=y_pred)
        return precision, recall


class LrTokenComparator(BaseLrComparator):
    def __init__(self, vectorizermodel='tfidf',
                 scoresuffix='tfidf',
                 on=None,
                 store_threshold=_tfidf__store_threshold_value,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 ngram_range=(1, 1),
                 stop_words=None,
                 strip_accents='ascii',
                 analyzer='word',
                 **kwargs):
        """

        Args:
            vectorizermodel (str): {'tfidf' or 'cv'} for TfIdfVectorizer or CountVectorizer
            scoresuffix (str):
            on (str):
            store_threshold (float): variable on which to store the threshold
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            ngram_range (tuple): default (1,1)
            stop_words (list) : {'english'}, list, or None (default)
            strip_accents (str): {'ascii', 'unicode', None}
            analyzer (str): {'word', 'char'} or callable. Whether the feature should be made of word or character n-grams.
        """
        BaseLrComparator.__init__(self,
                                  ixname=ixname,
                                  lsuffix=lsuffix,
                                  rsuffix=rsuffix,
                                  on=on,
                                  scoresuffix=scoresuffix,
                                  store_threshold=store_threshold
                                  )
        self._vocab = list()
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.strip_accents = strip_accents
        if vectorizermodel == 'tfidf':
            vectorizerclass = TfidfVectorizer
        elif vectorizermodel == 'cv':
            vectorizerclass = CountVectorizer
        else:
            raise ValueError('{} not in [cv, tfidf]'.format(vectorizermodel))
        self.tokenizer = vectorizerclass(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            strip_accents=self.strip_accents
        )
        self.store_threshold = store_threshold

    def fit(self, left, right, addvocab='add'):
        """
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):
            addvocab (str)
        Returns:
            self
        """
        newleft, newright = self._toseries(left=left, right=right)
        if addvocab in ['add', 'replace']:
            # TODO: Check
            try:
                _tempabbefore = len(self._vocab)
                _tempvocbefore = len(self.tokenizer.vocabulary_)
            except:
                _tempvocbefore = 0
                _tempabbefore = 0
            self._vocab = _update_vocab(left=newleft, right=newright, vocab=self._vocab, addvocab=addvocab)
            self.tokenizer = self.tokenizer.fit(self._vocab)
            _tempvocafter = len(self.tokenizer.vocabulary_)
            _tempabafter = len(self._vocab)
            print('vocab now: {}, {} lines added'.format(_tempabafter, _tempabafter - _tempabbefore))
            print('vocabulary_ now: {}, {} lines added'.format(_tempvocafter, _tempvocafter - _tempvocbefore))
        return self

    def transform(self, left, right, addvocab='add', *args, **kwargs):
        """
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):
            addvocab (str)
        Returns:
            pd.Series
        """
        newleft, newright = self._toseries(left=left, right=right)
        self.fit(left=newleft, right=newright, addvocab=addvocab)
        score = _transform_tkscore(
            left=newleft,
            right=newright,
            tokenizer=self.tokenizer,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            threshold=self.store_threshold,
            outcol=self.outcol
        )
        return score


class LrIdComparator(BaseLrComparator):
    def __init__(self,
                 scoresuffix='exact',
                 on=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 store_threshold=1,
                 **kwargs):
        """

        Args:
            tokenizer (str): 'tfidf', 'cv'
            threshold:
            **kwargs:
        """
        BaseLrComparator.__init__(self,
                                  ixname=ixname,
                                  lsuffix=lsuffix,
                                  rsuffix=rsuffix,
                                  on=on,
                                  scoresuffix=scoresuffix,
                                  store_threshold=store_threshold
                                  )

    def fit(self, left=None, right=None, *args, **kwargs):
        """
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):
        Returns:
            self
        """
        return self

    def transform(self, left, right, *args, **kwargs):
        """
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):
        Returns:
            pd.Series
        """
        newleft, newright = self._todf(left=left, right=right)
        score = pd.merge(
            left=newleft.reset_index(drop=False),
            right=newright.reset_index(drop=False),
            left_on=self.on,
            right_on=self.on,
            how='inner',
            suffixes=['_' + self.lsuffix, '_' + self.rsuffix]
        )
        score = score[self.ixnamepairs].set_index(self.ixnamepairs)
        score[self.outcol] = 1
        score = score[self.outcol]
        return score


class LrConnector:
    def __init__(self,
                 lrcomparators,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 pruning_thresholds=None):
        self.lrconnectors = lrcomparators
        self.outcols = [tk.outcol for tk in self.lrconnectors]
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft = '_'.join([self.ixname, self.lsuffix])
        self.ixnameright = '_'.join([self.ixname, self.rsuffix])
        self.ixnamepairs = [self.ixnameleft, self.ixnameright]
        if not pruning_thresholds is None:
            assert isinstance(pruning_thresholds, dict)
            self.pruning_thresholds = pruning_thresholds
        else:
            self.pruning_thresholds = dict()

        pass

    def transform(self, left, right, addvocab='add', verbose=False):
        scores = dict()
        ix_taken = pd.MultiIndex(levels=[[], []],
                                 labels=[[], []],
                                 names=self.ixnamepairs
                                 )
        for con in self.lrconnectors:
            assert isinstance(con, BaseLrComparator)
            transfo_score = con.transform(left=left, right=right, addvocab=addvocab)
            scores[con.outcol] = transfo_score
            assert transfo_score.index.names == ix_taken.names
            if self.pruning_thresholds.get(con.on) is not None:
                ix_taken = ix_taken.union(
                    transfo_score.loc[transfo_score >= self.pruning_thresholds[con.on]].index
                )
            del transfo_score

        X_scores = pd.DataFrame(index=ix_taken)
        for k in scores.keys():
            X_scores[k] = scores[k].loc[ix_taken.intersection(scores[k].index)]
            pass
        # case we have no id cols or text cols make a cartesian joi
        if X_scores.shape[1] == 0:
            X_scores = connectors.cartesian_join(left[[]], right[[]])
        if verbose is True:
            possiblepairs = left.shape[0] * right.shape[0]
            actualpairs = X_scores.shape[0]
            compression = int(possiblepairs / actualpairs)
            print(
                '{} | Pruning compression factor of {} on {} possibles pairs'.format(
                    pd.datetime.now(), compression, possiblepairs
                )
            )
        return X_scores

    def fit(self, *args, **kwargs):
        return self

    def _evalscore(self, left, right, y_true):
        # assert hasattr(self, 'transform') and callable(getattr(self, 'transform'))
        # noinspection
        y_pred = self.transform(left=left, right=right)
        precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=y_pred)
        return precision, recall


class LrDuplicateFinder:
    """
            score plan
        {
            name_col:{
                'type': one of ['FreeText', 'Category', 'Id', 'Code']
                'stop_words': [list of stopwords]
                'use_scores':
                    defaults for free text:
                        ['exact', 'fuzzy', 'tf-idf', 'n-gram_wostopwords' ,'token_wostopwords']
                    defaults for category:
                        ['exact'] --> but not taken into account for pruning
                    defaults for Id:
                        ['exact']
                    defaults for code: --> but not taken into account for pruning
                        ['exact', 'first n_digits/ngrams' --> to do]
                'threshold'
                    'tf-idf' --> Default 0.3
                    'exact' --> Default 1
            }
        }
    """

    def __init__(self,
                 scoreplan,
                 prefunc=None,
                 estimator=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
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
        self._ixnameleft = '_'.join([self.ixname, self.lsuffix])
        self._ixnameright = '_'.join([self.ixname, self.rsuffix])
        self._ixnamepairs = [self._ixnameleft, self._ixnameright]
        self._suffixascii = 'ascii'
        self._suffixwosw = 'wostopwords'
        self._suffixid = 'cleaned'
        self._suffixexact = 'exact'
        self._suffixtoken = 'token'
        self._suffixtfidf = 'tfidf'
        self._suffixfuzzy = 'fuzzy'
        self._suffixngram = 'ngram'
        # From score plan initiate the list of columns
        self._usedcols = scoreplan.keys()
        self._id_cols = list()
        self._code_cols = list()
        self._text_cols = list()
        self._cat_cols = list()
        self._connectcols = list()
        # Initiate the list of values used for the FreeTextAnalyszs
        self._stop_words = dict()
        self._pruning_thresholds = dict()
        self._lrcomparators = list()
        self._scorenames = dict()
        self._sbspipeplan = dict()
        self._ngram_char = (1, 2)
        self._ngram_word = (1, 1)
        self._vecmodel_cv = CountVectorizer
        self._vecmodel_tfidf = TfidfVectorizer
        self.verbose = verbose
        # Initiate for subclass the scikit-learn pipeline
        self._skmodel = BaseEstimator()
        # Loop through the score plan
        for inputfield in self._usedcols:
            self._initscoreplan(inputfield=inputfield)
        self.lrmodel = LrConnector(
            lrcomparators=self._lrcomparators,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            pruning_thresholds=self._pruning_thresholds
        )
        self._connectcols = [con.outcol for con in self._lrcomparators]
        # Prepare the Pipe
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

    def _initscoreplan(self, inputfield):
        scoretype = self.scoreplan[inputfield]['type']
        # inputfield: 'name'
        # actual field 'name_ascii' or 'duns_cleaned'
        # If the field is not free text
        if scoretype in ['Id', 'Category', 'Code']:
            actualcolname = '_'.join([inputfield, self._suffixid])
            self._scorenames[inputfield] = ['_'.join([actualcolname, self._suffixexact])]
            if scoretype == 'Id':
                # This score is calculated via Left-Right Id Comparator
                self._id_cols.append(actualcolname)
                self._lrcomparators.append(LrIdComparator(
                    on=actualcolname,
                    ixname=self.ixname,
                    lsuffix=self.lsuffix,
                    rsuffix=self.rsuffix,
                    scoresuffix=self._suffixexact
                ))
            else:
                self._sbspipeplan[actualcolname] = [self._suffixexact]
                if scoretype == 'Category':
                    self._cat_cols.append(actualcolname)
                elif scoretype == 'Code':
                    self._code_cols.append(actualcolname)
                    self._scorenames[inputfield].append('_'.join([actualcolname, self._suffixngram]))
                    self._lrcomparators.append(LrTokenComparator(
                        on=actualcolname,
                        ixname=self.ixname,
                        lsuffix=self.lsuffix,
                        rsuffix=self.rsuffix,
                        scoresuffix=self._suffixexact,
                        vectorizermodel='cv',
                        ngram_range=self._ngram_char,
                        analyzer='char'
                    ))
        elif scoretype == 'FreeText':
            actualcolname = '_'.join([inputfield, self._suffixascii])
            self._text_cols.append(actualcolname)
            self._scorenames[inputfield] = list()
            if self.scoreplan[inputfield].get('stop_words') is not None:
                self._stop_words[actualcolname] = self.scoreplan[inputfield].get('stop_words')
            if self.scoreplan[inputfield].get('threshold') is not None:
                assert self.scoreplan[inputfield].get('threshold') <= 1.0
            self._pruning_thresholds[actualcolname] = self.scoreplan[inputfield].get('threshold')
            if self.scoreplan[inputfield].get('use_scores') is None:
                use_scores = [
                    self._suffixtfidf,
                    self._suffixngram,
                    self._suffixfuzzy,
                    self._suffixtoken
                ]
            else:
                use_scores = self.scoreplan[inputfield].get('use_scores')
            # Loop through use_scores
            for s2 in use_scores:
                self._scorenames[inputfield].append(
                    '_'.join([actualcolname, s2])
                )
                if s2 == self._suffixtfidf:
                    self._lrcomparators.append(LrTokenComparator(
                        on=actualcolname,
                        ixname=self.ixname,
                        lsuffix=self.lsuffix,
                        rsuffix=self.rsuffix,
                        scoresuffix=self._suffixtfidf,
                        vectorizermodel='tfidf',
                        ngram_range=self._ngram_word,
                        analyzer='word',
                        stop_words=self._stop_words.get(actualcolname)
                    ))
                elif s2 == self._suffixngram:
                    self._lrcomparators.append(LrTokenComparator(
                        on=actualcolname,
                        ixname=self.ixname,
                        lsuffix=self.lsuffix,
                        rsuffix=self.rsuffix,
                        scoresuffix=self._suffixngram,
                        vectorizermodel='tfidf',
                        ngram_range=self._ngram_char,
                        analyzer='char',
                        stop_words=self._stop_words.get(actualcolname)
                    ))
                elif s2 in [self._suffixtoken, self._suffixfuzzy]:
                    if self._sbspipeplan.get(actualcolname) is None:
                        self._sbspipeplan[actualcolname] = [s2]
                    else:
                        self._sbspipeplan[actualcolname].append(s2)

                if self._stop_words.get(actualcolname) is not None:
                    swcolname = actualcolname + '_' + self._suffixwosw
                    self._scorenames[inputfield].append(
                        '_'.join([swcolname, self._suffixngram])
                    )
                    self._scorenames[inputfield].append(
                        '_'.join([swcolname, self._suffixtoken])
                    )
                    self._lrcomparators.append(LrTokenComparator(
                        on=swcolname,
                        ixname=self.ixname,
                        lsuffix=self.lsuffix,
                        rsuffix=self.rsuffix,
                        scoresuffix=self._suffixngram,
                        vectorizermodel='tfidf',
                        ngram_range=self._ngram_char,
                        analyzer='char'
                    ))
                    self._sbspipeplan[actualcolname + '_' + self._suffixwosw] = [self._suffixtoken]

        pass

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

    def _pruning_fit_transform(self, left, right, addvocab='add', verbose=False):
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
        X_scores = self.lrmodel.transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
        assert set(X_scores.columns.tolist()) == set(self._connectcols)
        return X_scores

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
        X_connect = self._pruning_fit_transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
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

        X_sbs = self._connectscores(left=newleft, right=newright, addvocab=addvocab, verbose=True)
        if X_sbs.shape[0] == 0:
            raise Warning(
                'No possible matches based on current pruning --> not possible to fit.',
                '\n - Provide better training data',
                '\n - Check the pruning threshold and ids',
            )

        if verbose:
            precision, recall = _evalprecisionrecall(y_true=y_train, y_pred=X_sbs)
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
        self._pruning_fit_transform(left=newleft, right=newright, addvocab=addvocab)
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

        self._pruning_fit_transform(left=newleft, right=newright, addvocab=addvocab)
        X_sbs = self._connectscores(left=newleft, right=newright, addvocab=addvocab, verbose=verbose)
        precision, recall = _evalprecisionrecall(y_true=y_train, y_pred=X_sbs)
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


def _update_vocab(left, right, vocab=None, addvocab='add'):
    """

    Args:
        vocab (list):
        addvocab (str):
            - 'add' --> add new vocab to existing vocab
            - 'keep' --> keep existing vocab
            - 'replace' --> replace existing vocab with new
    Returns:
        list
    """
    assert isinstance(left, pd.Series)
    assert isinstance(right, pd.Series)
    assert addvocab in ['add', 'keep', 'replace']
    if vocab is None:
        vocab = list()
    assert isinstance(vocab, list)
    if addvocab == 'keep':
        return vocab
    else:
        left = left.dropna().copy().tolist()
        right = right.dropna().copy().tolist()
        vocab2 = left + right
        if addvocab == 'add':
            vocab += vocab2
        elif addvocab == 'replace':
            vocab = vocab2
    return vocab


def _fit_tokenizer(left, right, tokenizer, vocab=None, addvocab='add'):
    """
    update the vocabulary of the tokenizer
    Args:
        left (pd.Series):
        right (pd.Series):
        vocab (list)
        addvocab (str):


    Returns:
        self
    """

    vocab = _update_vocab(left=left, right=right, vocab=vocab, addvocab=addvocab)
    tokenizer.fit(vocab)
    return tokenizer


def _transform_tkscore(left,
                       right,
                       tokenizer,
                       ixname='ix',
                       lsuffix='left',
                       rsuffix='right',
                       outcol='score',
                       threshold=None):
    """
    DO NOT RE-FIT the algo
    Args:
        left (pd.Series):
        right (pd.Series):
        outcol (str):
    Returns:
        pd.Series
    """
    scorename = outcol
    left = left.dropna().copy()
    right = right.dropna().copy()
    assert isinstance(left, pd.Series)
    assert isinstance(right, pd.Series)
    ixnameleft = ixname + '_' + lsuffix
    ixnameright = ixname + '_' + rsuffix
    ixnamepairs = [ixnameleft, ixnameright]
    # If we cannot find a single value we return blank
    if left.shape[0] == 0 or right.shape[0] == 0:
        ix = pd.MultiIndex(levels=[[], []],
                           labels=[[], []],
                           names=ixnamepairs
                           )
        r = pd.Series(index=ix, name=scorename)
        return r
    left_tfidf = tokenizer.transform(left)
    right_tfidf = tokenizer.transform(right)
    X = pd.DataFrame(
        cosine_similarity(left_tfidf, right_tfidf),
        columns=right.index
    )
    X[ixnameleft] = left.index
    score = pd.melt(
        X,
        id_vars=ixnameleft,
        var_name=ixnameright,
        value_name=scorename
    ).set_index(
        ixnamepairs
    )
    if not threshold is None:
        score = score[score[scorename] >= threshold]
    score = score[scorename]
    return score
