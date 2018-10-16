import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

from wookie.connectors import cartesian_join, createsbs, indexwithytrue
from wookie.preutils import lowerascii, idtostr, rmvstopwords, _suffixexact, _suffixtoken, _suffixfuzzy
# noinspection PyProtectedMember
from wookie.sbscomparators import _evalprecisionrecall, _metrics, DataPasser, PipeSbsComparator

_tfidf_store_threshold_value = 0.5


class BaseLrComparator(TransformerMixin):
    """
    This is the base Left-Right Comparator
    Idea is that is should have take a left dataframe, a right dataframe,
    and return a combination of two, with a comparison score
    """

    def __init__(self,
                 on=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 scoresuffix='score',
                 store_threshold=0.0
                 ):
        """

        Args:
            on(str): column to use on the left and right df
            ixname (str): name of the index of left and right
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): score suffix: the outputvector has the name on + '_' + scoresuffix
            store_threshold (float): threshold to use to store the relevance score
        """
        TransformerMixin.__init__(self)
        if on is None:
            on = 'none'
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
        convert to dataframe with one column withoutnulls and copy
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):

        Returns:
            pd.DataFrame, pd.DataFrame
        """
        newleft = pd.DataFrame()
        newright = pd.DataFrame()
        if isinstance(left, pd.DataFrame):
            if self.on is not None and self.on != 'none':
                newleft = left[[self.on]].dropna(subset=[self.on]).copy()
            else:
                newleft = left.copy()
        if isinstance(right, pd.DataFrame):
            if self.on is not None and self.on != 'none':
                newright = right[[self.on]].dropna(subset=[self.on]).copy()
            else:
                newright = right.copy()

        if isinstance(left, pd.Series):
            newleft = pd.DataFrame(left.dropna().copy())
        if isinstance(right, pd.Series):
            newright = pd.DataFrame(right.dropna().copy())
        for s, c in zip(['left', 'right'], [left, right]):
            if not isinstance(c, pd.Series) and not isinstance(c, pd.DataFrame):
                raise TypeError('type {} not Series or DataFrame for side {}'.format(type(c), s))
        return newleft, newright

    def evalscore(self, left, right, y_true):
        """
        evaluate precision and recall
        Args:
            left (pd.DataFrame/pd.Series):
            right (pd.DataFrame/pd.Series):
            y_true (pd.Series):

        Returns:
            float, float: precision and recall
        """
        # assert hasattr(self, 'transform') and callable(getattr(self, 'transform'))
        # noinspection
        y_pred = self.transform(left=left, right=right)
        precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=y_pred)
        return precision, recall


class LrTokenComparator(BaseLrComparator):
    def __init__(self, vectorizermodel='tfidf',
                 scoresuffix='tfidf',
                 on=None,
                 store_threshold=_tfidf_store_threshold_value,
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
            on (str): column to compare
            store_threshold (float): variable above which the similarity score is stored
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            ngram_range (tuple): default (1,1)
            stop_words (list) : {'english'}, list, or None (default)
            strip_accents (str): {'ascii', 'unicode', None}
            analyzer (str): {'word', 'char'} or callable. Whether the feature should be made of word or chars n-grams.
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
        self.vectorizer = vectorizerclass(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            strip_accents=self.strip_accents
        )
        self.store_threshold = store_threshold

    def fit(self, left=None, right=None):
        """
        Do Nothing
        Args:
            left (pd.DataFrame/pd.Series):
            right (pd.DataFrame/pd.Series):

        Returns:

        """

        # DO NOTHING
        return self

    def transform(self, left, right, addvocab='add', *args, **kwargs):
        """
        Add, keep, or replace new vocabulary to the vectorizer
        Fit the tokenizer with the new vocabulary
        Calculate the cosine_simlarity score for the left and right columns \
            using the output of the vectorizer
        Args:
            left (pd.Series/pd.DataFrame):
            right (pd.Series/pd.DataFrame):
            addvocab (str): in ['add', 'keep', 'replace']
                'add' --> add new vocabulary to current
                'keep' --> only keep current vocabulary
                'replace' --> replace current vocabulary with new
        Returns:
            pd.Series : {['ix_left', 'ix_right']: 'name_tfidf'}
        """
        newleft, newright = self._toseries(left=left, right=right)
        # Fit
        if addvocab in ['add', 'replace']:
            self._vocab = _update_vocab(left=newleft, right=newright, vocab=self._vocab, addvocab=addvocab)
            self.vectorizer = self.vectorizer.fit(self._vocab)
        score = _transform_tkscore(
            left=newleft,
            right=newright,
            vectorizer=self.vectorizer,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            threshold=self.store_threshold,
            outcol=self.outcol
        )
        return score


class LrIdComparator(BaseLrComparator):
    def __init__(self,
                 on,
                 scoresuffix='exact',
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 store_threshold=1.0,
                 **kwargs):
        """

        Args:
            on (str): column to compare
            scoresuffix (str): name of the suffix added to the column name for the score name
            ixname: 'ix'
            lsuffix (str): 'left'
            rsuffix (str): 'right'
            store_threshold(flat): variable above which the similarity score is stored
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
        # Do nothing
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
            left (pd.Series/pd.DataFrame): {'ix':['duns', ...]}
            right (pd.Series/pd.DataFrame):
        Returns:
            pd.Series: {['ix_left', 'ix_right']: 'duns_exact'}
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


class LrPruningConnector:
    """
    This class pipes together multiple instances of LrComparator: LrTokenComparator or LrIdComparator
    Returns a DataFrame showing the similarity score of left records vs right records
    If all similarity scores are below the pruning threshold:
        the pair is dropped
    If any similarity score is above the pruning threshold:
        it is kept (for example to be ingested to a Side-By-Side Comparator and Scikit-Learn Estimator)
    """

    def __init__(self,
                 lrcomparators,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 pruning_thresholds=None):
        """
        Not on the pruning_threshold:
            if the value is None, that means you do not use this similarity score as the basis for further analysis

        Args:
            lrcomparators (list): list of LrComparator
            ixname (str): 'ix'
            lsuffix (str): 'left'
            rsuffix (str): 'right'
            pruning_thresholds (dict): {'name':0.6, 'city':None}
        Examples:
            pruning_thresholds: {'name':0.6, 'city':None}: only pairs with similarity on name >= 0.6 will be kept
            pruning_thresholds: {'name':0.6, 'city':05}: only pairs with similarity on name >= 0.6 \
                or similarity on 'city' >=0.5 will be kept

        """
        self.lrconnectors = lrcomparators
        self.outcols = [tk.outcol for tk in self.lrconnectors]
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft = '_'.join([self.ixname, self.lsuffix])
        self.ixnameright = '_'.join([self.ixname, self.rsuffix])
        self.ixnamepairs = [self.ixnameleft, self.ixnameright]
        if pruning_thresholds is not None:
            assert isinstance(pruning_thresholds, dict)
            self.pruning_thresholds = pruning_thresholds
        else:
            self.pruning_thresholds = dict()

        pass

    def transform(self, left, right, addvocab='add', verbose=False):
        """
        # return all the similarity scores of each LR Comparator as a dataframe
        # In case we have no id cols or text cols make a cartesian join
        Args:
            left (pd.DataFrame): {'ix' :['name', 'duns']}
            right (pd.DataFrame):
            addvocab (str): add, keep, or replace
            verbose (bool): print the compression factor (how many pairs selected out of possible)

        Returns:
            pd.DataFrame : {['ix_left', 'ix_right']: ['name_tfidf', 'duns_exact'...}
        """
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
        # case we have no id cols or text cols make a cartesian join
        if X_scores.shape[1] == 0:
            X_scores = cartesian_join(left[[]], right[[]])
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
        """
        DO NOTHING
        Args:
            *args:
            **kwargs:

        Returns:

        """
        return self

    def evalscore(self, left, right, y_true, verbose=False):
        """
        Evaluate precision and recall
        Args:
            left (pd.DataFrame): {'ix':[cols..]}
            right (pd.DataFrame):
            y_true (pd.Series): {['ix_left', 'ix_right']: 'y_true'}
            verbose (bool): print out precision and recall

        Returns:
            float, float: precision and recall
        """
        # assert hasattr(self, 'transform') and callable(getattr(self, 'transform'))
        # noinspection
        y_pred = self.transform(left=left, right=right)
        precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=y_pred)
        if verbose:
            print(
                '{} | LrPruningConnector score: precision: {:.2%}, recall: {:.2%}'.format(
                    pd.datetime.now(),
                    precision,
                    recall
                )
            )
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
                'pruning_threshold'
                        0.6
                        pruning threshold to pass to the LrPruningConnector
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
            scoreplan (dict):
            prefunc (callable): preprocessing function. Add features to the data. Default lambda df: df
            estimator (BaseEstimator): Sklearn Estimator, Default RandomForest()
            ixname (str): 'ix'
            lsuffix (str): 'left'
            rsuffix (str): 'right'
            verbose (bool)
        Examples:
                dedupe = wookie.lrcomparators.LrDuplicateFinder(
                    prefunc=preprocessing.preparedf,
                    scoreplan={
                        'name': {
                            'type': 'FreeText',
                            'stop_words': preprocessing.companystopwords,
                            'use_scores': ['tfidf', 'ngram'],
                            'threshold': 0.6,
                        }
                        'street': {
                            'type': 'FreeText',
                            'stop_words': preprocessing.streetstopwords,
                            'use_scores': ['tfidf', 'ngram', 'token'],
                            'threshold': 0.6
                        },
                        'city': {
                            'type': 'FreeText',
                            'stop_words': preprocessing.citystopwords,
                            'use_scores': ['tfidf', 'ngram', 'fuzzy'],
                            'threshold': None
                        },
                        'duns': {'type': 'Id'},
                        'postalcode': {'type': 'Code'},
                        'countrycode': {'type': 'Category'}
                    },
                    estimator=GradientBoostingClassifier()
                )
        """
        if estimator is not None:
            assert issubclass(type(estimator), BaseEstimator)
            self.estimator = estimator
        else:
            self.estimator = RandomForestClassifier()

        if prefunc is not None:
            assert callable(prefunc)
        else:
            def passdata(df):
                return df

            prefunc = passdata
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
        self._suffixexact = _suffixexact
        self._suffixtoken = _suffixtoken
        self._suffixtfidf = 'tfidf'
        self._suffixfuzzy = _suffixfuzzy
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
        self.lrmodel = LrPruningConnector(
            lrcomparators=self._lrcomparators,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            pruning_thresholds=self._pruning_thresholds
        )
        self._connectcols = [con.outcol for con in self._lrcomparators]
        self._sbscols = list(self._sbspipeplan.keys())
        # Prepare the Pipe
        dp_connectscores = DataPasser(on_cols=self._connectcols)

        sbspipe = PipeSbsComparator(
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
        """
        Routine to initiate the class
        Dig into the code to understand.
        Args:
            inputfield (str):

        Returns:
            None
        """
        scoretype = self.scoreplan[inputfield]['type']
        # inputfield: 'name'
        # actualcolname 'name_ascii' or 'duns_cleaned'
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
                if self.scoreplan[inputfield].get('threshold') is not None:
                    self._pruning_thresholds[actualcolname] = self.scoreplan[inputfield].get('threshold')
                else:
                    self._pruning_thresholds[actualcolname] = 1.0
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
        - Pass the input data (left or right) through a preprocessing function prefunc
        - add normalized columns
            for Free text: 'name' --> 'name_ascii', 'name_ascii_wostopwords' if stop words are passed
            For others: 'duns' --> 'duns_cleaned' with passing idtostr method
        Args:
            df (pd.DataFrame):

        Returns:
            new (pd.DataFrame)
        """
        new = df.copy()
        new = self.prefunc(new)
        for inputfield in self.scoreplan.keys():
            scoretype = self.scoreplan[inputfield]['type']
            if scoretype == 'FreeText':
                actualcolname = '_'.join([inputfield, self._suffixascii])
                new[actualcolname] = new[inputfield].apply(lowerascii)
                if actualcolname in self._stop_words.keys():
                    colnamewosw = '_'.join([actualcolname, self._suffixwosw])
                    new[colnamewosw] = new[actualcolname].apply(
                        lambda r: rmvstopwords(r, stop_words=self._stop_words[actualcolname])
                    )
            else:
                actualcolname = '_'.join([inputfield, self._suffixid])
                new[actualcolname] = new[inputfield].apply(
                    lowerascii
                ).apply(
                    idtostr
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
            verbose (bool)

        Returns:
            pd.DataFrame {[ix_left, ix_right]: [scores]}
        """
        X_scores = self.lrmodel.transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
        assert set(X_scores.columns.tolist()) == set(self._connectcols)
        return X_scores

    def _lr_to_sbs(self, left, right, addvocab='add', verbose=False):
        """
        Return a side by side analysis of the data
        Args:
            left: {ix: [cols]}
            right: {ix: [cols]}
            addvocab (str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new
        Returns:
            pd.DataFrame:  {[ix_left, ix_right]: [scores, 'name_left', 'name_right']}
        """
        X_scores = self.lrmodel.transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
        assert set(X_scores.columns.tolist()) == set(self._connectcols)
        X_sbs = createsbs(
            pairs=X_scores,
            left=left,
            right=right,
            use_cols=self._sbscols,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        return X_sbs

    def fit(self, left, right, y_true, addvocab='add', verbose=False):
        """
        # Plan
        - prepare the data
        - Use the LrPruningConnector to create a Side by Side view of the records with similarity scores
        - Use the SbsPipeComparator to do further analysis (Levenshtein) on those side by side
        - Fit an estimator on the results
        # Exception:
        - if no matches are found during pruning - raise warning
        - The estimator is only trained on the output from the pruning step, some matches may be already missed

        Args:
            left (pd.DataFrame): {'ix':[name, duns...]}
            right (pd.DataFrame):
            y_true (pd.Series/pd.DataFrame): pairs {['ix_left', 'ix_right']: y_true}
            addvocab (str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new
            verbose (bool):

        Returns:
            self
        """
        if verbose:
            print('{} | Start fit'.format(pd.datetime.now()))
        newleft = self._prepare_data(left)
        newright = self._prepare_data(right)

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true['y_true']

        X_sbs = self._lr_to_sbs(left=newleft, right=newright, addvocab=addvocab, verbose=verbose)
        if X_sbs.shape[0] == 0:
            raise Warning(
                'No possible matches based on current pruning --> not possible to fit.',
                '\n - Provide better training data',
                '\n - Check the pruning threshold and ids',
            )

        if verbose:
            precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=X_sbs)
            print('{} | Pruning score: precision: {:.2%}, recall: {:.2%}'.format(pd.datetime.now(), precision, recall))

        # Redefine X_sbs and y_true to have a common index
        ix_common = y_true.index.intersection(
            X_sbs.index
        )
        y_true = y_true.loc[ix_common]
        X_sbs = X_sbs.loc[ix_common]

        # Fit the ScikitLearn Model
        self._skmodel.fit(X_sbs, y_true)

        if verbose:
            scores = self.score(left=left, right=right, pairs=y_true, kind='all')
            assert isinstance(scores, dict)
            print(
                '{} | Estimator score: precision: {:.2%}, recall: {:.2%}'.format(
                    pd.datetime.now(),
                    scores['precision'],
                    scores['recall']
                )
            )
            pass
        return self

    def _pruning_pred(self, left, right, y_true, addvocab='add'):
        """
        Return the result of the pruning step with all possible matches marked as positive match
        For precision and recall calculation
        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            y_true (pd.Series/pd.DataFrame): {['ix_left', 'ix_right']: y_true}
            addvocab (str):

        Returns:
            pd.Series {['ix_left', 'ix_right']: 1}
        """
        newleft = self._prepare_data(left)
        newright = self._prepare_data(right)
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true['y_true']
        X_sbs = self._lr_to_sbs(left=newleft, right=newright, addvocab=addvocab)
        y_pred = pd.Series(index=X_sbs.index).fillna(1)
        y_pred = indexwithytrue(y_true=y_true, y_pred=y_pred)
        return y_pred

    def predict_proba(self, left, right, addvocab='add', verbose=False, addmissingleft=False) -> pd.DataFrame:
        """
        Predict_proba
        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            addvocab (pd.DataFrame):
            verbose (bool):
            addmissingleft (bool): add the missing left values where no possible matches found during pruning

        Returns:
            pd.DataFrame : {['ix_left', 'ix_right']: [0, 1]}

        Examples:
            if addmissingleft is True:

            ix_left     ix_right    0       1
            m93         m51         0.8     0.2
            m93         m87         0.4     0.6
            m16         None        1.0     0.0

            In this example, m93 has two possible matches from the pruning step (m51 and m87)
            Of those two, only one (m51) is a predicted match
            m16 has no possible matches because all of the possible records from the right record were pruned out
        """

        if verbose:
            print('{} | Start pred'.format(pd.datetime.now()))
        newleft = self._prepare_data(left)
        newright = self._prepare_data(right)

        X_sbs = self._lr_to_sbs(left=newleft, right=newright, addvocab=addvocab, verbose=verbose)

        # if we have results
        if X_sbs.shape[0] > 0:
            df_pred = self._skmodel.predict_proba(X_sbs)
            df_pred = pd.DataFrame(
                df_pred,
                index=X_sbs.index
            )
        else:
            # Create an empty recipient
            df_pred = pd.DataFrame(
                index=pd.MultiIndex(
                    levels=[[], []],
                    labels=[[], []],
                    names=self._ixnamepairs
                ),
                columns=[0, 1]
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
            missing_lefts = pd.DataFrame(index=missing_lefts, columns=[0, 1])
            # Because those matches were not found: 100% probability of not being a match, 0% proba of being a match
            missing_lefts[0] = 1
            missing_lefts[1] = 0
            df_pred = pd.concat([df_pred, missing_lefts], axis=0, ignore_index=False)

        assert isinstance(df_pred, pd.DataFrame)
        assert set(df_pred.columns) == {0, 1}
        assert df_pred.index.names == self._ixnamepairs
        return df_pred

    def predict(self, left, right, addvocab='add', verbose=False, addmissingleft=False):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            addvocab(str):
                - 'add' --> add new vocab to existing vocab
                - 'keep' --> keep existing vocab
                - 'replace' --> replace existing vocab with new,
            verbose (bool)
            addmissingleft (bool): see doc of predict_proba

        Returns:
            pd.Series: {['ix_left', 'ix_right']: [0.0 or 1.0]), 1.0 being a match
        """
        df_pred = self.predict_proba(left=left, right=right, addvocab=addvocab,
                                     verbose=verbose, addmissingleft=addmissingleft)
        # noinspection PyUnresolvedReferences
        y_pred = (df_pred[1] > 0.5).astype(float)
        assert isinstance(y_pred, pd.Series)
        y_pred.name = 'y_pred'
        return y_pred

    def _evalpruning(self, left, right, y_true, addvocab='add', verbose=False):
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true['y_true']

        X_sbs = self._lr_to_sbs(left=newleft, right=newright, addvocab=addvocab, verbose=verbose)
        precision, recall = _evalprecisionrecall(y_true=y_true, y_pred=X_sbs)
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
        scores = self._scores(left=left, right=right, y_true=pairs)
        if kind != 'all':
            return scores[kind]
        else:
            return scores

    def _scores(self, left, right, y_true):
        """

        Args:
            left:
            right:
            y_true:

        Returns:
            dict
        """
        y_pred = self.predict(left=left, right=right)
        assert isinstance(y_pred, pd.Series)
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true['y_true']
        y_true = y_true
        assert isinstance(y_true, pd.Series)
        scores = _metrics(y_true=y_true, y_pred=y_pred)
        return scores


def _update_vocab(left, right, vocab=None, addvocab='add'):
    """

    Args:
        left (pd.Series)
        right (pd.Series)
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


def _transform_tkscore(left,
                       right,
                       vectorizer,
                       ixname='ix',
                       lsuffix='left',
                       rsuffix='right',
                       outcol='score',
                       threshold=None):
    """
    Args:
        left (pd.Series):
        right (pd.Series):
        vectorizer: TfIdf vectorizer or CountVectorizer
        outcol (str): name of the output series
        threshold (float): Filter on scores greater than or equal to the threshold
        ixname (str)
        lsuffix (str)
        rsuffix(str)
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
    left_tfidf = vectorizer.transform(left)
    right_tfidf = vectorizer.transform(right)
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
    if threshold is not None:
        score = score[score[scorename] >= threshold]
    score = score[scorename]
    return score
