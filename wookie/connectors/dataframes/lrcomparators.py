from copy import deepcopy

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import Imputer

from wookie import createsbs, LrExactComparator
from wookie.base import BaseLrComparator
from wookie.comparators.leftright.tokenizers import LrTokenComparator
from wookie.connectors.dataframes.base import indexwithytrue, metrics, evalprecisionrecall, cartesian_join
from wookie.preutils import lowerascii, idtostr, rmvstopwords, datapasser, \
    suffixexact, suffixtoken, suffixfuzzy, concatixnames, \
    name_freetext, name_exact, name_pruning_threshold, \
    name_stop_words, name_usescores

suffix_ngram = 'ngram'
suffix_tfdif = 'tfidf'

# noinspection PyProtectedMember
from wookie.sbscomparators import DataPasser, PipeSbsComparator


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
                 scoreplan,
                 prefunc=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right'):
        """
        Not on the pruning_threshold:
            if the value is None, that means you do not use this similarity score as the basis for further analysis

        Args:
            scoreplan (dict): list of LrComparator,
            prefunc (callable)
            ixname (str): 'ix'
            lsuffix (str): 'left'
            rsuffix (str): 'right'
        Examples:
            pruning_thresholds: {'name':0.6, 'city':None}: only pairs with similarity on name >= 0.6 will be kept
            pruning_thresholds: {'name':0.6, 'city':05}: only pairs with similarity on name >= 0.6 \
                or similarity on 'city' >=0.5 will be kept

        """
        if prefunc is not None:
            assert callable(prefunc)
        else:
            prefunc = datapasser
        self.prefunc = prefunc
        self.scoreplan = deepcopy(scoreplan)
        # Define column names
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix
        )
        self._suffixascii = 'ascii'
        self._suffixwosw = 'wostopwords'
        self._suffixid = 'cleaned'
        self._suffixexact = suffixexact
        self._suffixtoken = suffixtoken
        self.suffix_tfidf = suffix_tfdif
        self.suffix_ngram = suffix_ngram
        self._suffixfuzzy = suffixfuzzy

        # From score plan initiate the list of columns
        self.cols_used = scoreplan.keys()
        self.cols_exact = list()
        self.cols_freetext = list()
        self.outcols = list()
        # Initiate the list of values used for the LrConnectors
        self.stop_words = dict()
        self.pruning_thresholds = dict()
        self._lrcomparators = list()
        self._scorenames = dict()
        self._ngram_char = (1, 2)
        self._ngram_word = (1, 1)
        self._vecmodel_cv = CountVectorizer
        self._vecmodel_tfidf = TfidfVectorizer

        for inputfield in self.cols_used:
            self._initscoreplan(actualcolname=inputfield)

        self.outcols = [tk.outcol for tk in self._lrcomparators]

        pass

    def _initscoreplan(self, actualcolname):
        """
        Routine to initiate the class
        Dig into the code to understand.
        Args:
            actualcolname (str):

        Returns:
            None
        """
        scoretype = self.scoreplan[actualcolname]['type']
        # inputfield: 'name'
        # actualcolname 'name_ascii' or 'duns_cleaned'
        # If the field is not free text

        if scoretype == name_exact:
            self._scorenames[actualcolname] = ['_'.join([actualcolname, self._suffixexact])]
            # This score is calculated via Left-Right Id Comparator
            self.cols_exact.append(actualcolname)
            self._lrcomparators.append(LrExactComparator(
                on=actualcolname,
                ixname=self.ixname,
                lsuffix=self.lsuffix,
                rsuffix=self.rsuffix,
                scoresuffix=self._suffixexact
            ))
            if self.scoreplan[actualcolname].get(name_pruning_threshold) is not None:
                self.pruning_thresholds[actualcolname] = self.scoreplan[actualcolname].get(
                    name_pruning_threshold)
        elif scoretype == name_freetext:
            self.cols_freetext.append(actualcolname)
            self._scorenames[actualcolname] = list()
            if self.scoreplan[actualcolname].get(name_stop_words) is not None:
                self.stop_words[actualcolname] = self.scoreplan[actualcolname].get(name_stop_words)
            if self.scoreplan[actualcolname].get(name_pruning_threshold) is not None:
                self.pruning_thresholds[actualcolname] = self.scoreplan[actualcolname].get(name_pruning_threshold)
            if self.scoreplan[actualcolname].get(name_usescores) is None:
                use_scores = [
                    self.suffix_tfidf,
                    self.suffix_ngram
                ]
            else:
                use_scores = self.scoreplan[actualcolname].get(name_usescores)
            # Loop through use_scores
            for s2 in use_scores:
                self._scorenames[actualcolname].append(
                    '_'.join([actualcolname, s2])
                )
                if s2 == self.suffix_tfidf:
                    self._lrcomparators.append(LrTokenComparator(
                        on=actualcolname,
                        ixname=self.ixname,
                        lsuffix=self.lsuffix,
                        rsuffix=self.rsuffix,
                        scoresuffix=self.suffix_tfidf,
                        vectorizermodel='tfidf',
                        ngram_range=self._ngram_word,
                        analyzer='word',
                        stop_words=self.stop_words.get(actualcolname)
                    ))
                elif s2 == self.suffix_ngram:
                    self._lrcomparators.append(LrTokenComparator(
                        on=actualcolname,
                        ixname=self.ixname,
                        lsuffix=self.lsuffix,
                        rsuffix=self.rsuffix,
                        scoresuffix=self.suffix_ngram,
                        vectorizermodel='tfidf',
                        ngram_range=self._ngram_char,
                        analyzer='char',
                        stop_words=self.stop_words.get(actualcolname)
                    ))

    def fit_transform(self, left, right):
        self.fit()
        X_scores = self.transform(left=left, right=right)
        return X_scores

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
        for lrcomparator in self._lrcomparators:
            assert isinstance(lrcomparator, BaseLrComparator)
            transfo_score = lrcomparator.transform(left=left, right=right, addvocab=addvocab)
            scores[lrcomparator.outcol] = transfo_score
            assert transfo_score.index.names == ix_taken.names
            v_pairs_after = None
            v_newpairs = None
            v_possiblepairs = None
            v_pairs_matched = None
            v_pairs_before = None

            if verbose is True:
                v_pairs_before = len(ix_taken)
            if self.pruning_thresholds.get(lrcomparator.on) is not None:
                v_pairs_matched = len(transfo_score.loc[transfo_score >= self.pruning_thresholds.get(lrcomparator.on)])
                ix_taken = ix_taken.union(
                    transfo_score.loc[transfo_score >= self.pruning_thresholds[lrcomparator.on]].index
                )
            else:
                v_pairs_matched = 0

            if verbose is True:
                v_pairs_after = len(ix_taken)
                v_newpairs = v_pairs_after - v_pairs_before
                v_pairs_before = v_pairs_after
                v_possiblepairs = left.shape[0] * right.shape[0]
                print(
                    '{} | for score {}_{} with threshold of {}: pairs over ths: {:.2%}, new pairs added: {:.2%}'.format(
                        pd.datetime.now(),
                        lrcomparator.on,
                        lrcomparator.scoresuffix,
                        self.pruning_thresholds.get(lrcomparator.on),
                        v_pairs_matched / v_possiblepairs,
                        v_newpairs / v_possiblepairs
                    ))

            del transfo_score
        X_scores = pd.DataFrame(index=ix_taken)
        for k in scores.keys():
            X_scores[k] = scores[k].loc[ix_taken.intersection(scores[k].index)]
            pass
        # case we have no id cols or text cols make a cartesian join
        if X_scores.shape[1] == 0:
            X_scores = cartesian_join(left[[]], right[[]])
        if verbose is True:
            v_possiblepairs = left.shape[0] * right.shape[0]
            actualpairs = X_scores.shape[0]
            compression = int(v_possiblepairs / actualpairs)
            print(
                '{} | Pruning compression factor of {} on {} possibles pairs\n'.format(
                    pd.datetime.now(), compression, v_possiblepairs
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
        precision, recall = evalprecisionrecall(y_true=y_true, y_pred=y_pred)
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
                 classifier=None,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 verbose=False,
                 n_jobs=1
                 ):
        """
        Args:
            scoreplan (dict):
            prefunc (callable): preprocessing function. Add features to the data. Default lambda df: df
            classifier (BaseEstimator): Sklearn Estimator Classifier, Default RandomForest()
            ixname (str): 'ix'
            lsuffix (str): 'left'
            rsuffix (str): 'right'
            verbose (bool)
            n_jobs (int)
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
        if prefunc is not None:
            assert callable(prefunc)
        else:
            prefunc = datapasser
        self.prefunc = prefunc
        assert isinstance(scoreplan, dict)
        self.scoreplan = deepcopy(scoreplan)
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.n_jobs = n_jobs
        self.verbose = verbose

        if classifier is not None:
            assert issubclass(type(classifier), BaseEstimator)
            self.estimator = classifier
        else:
            self.estimator = RandomForestClassifier(n_jobs=self.n_jobs)

        # Derive the new column names
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix
        )
        self._suffixascii = 'ascii'
        self._suffixwosw = 'wostopwords'
        self._suffixid = 'cleaned'
        self._suffixexact = suffixexact
        self._suffixtoken = suffixtoken

        self._suffixfuzzy = suffixfuzzy

        # Enrich the score plan to pass on to the lrmodel
        self._usedcols = list(self.scoreplan.keys())
        self.stop_words = dict()
        self.scoreplan_lr = dict()
        self.scoreplan_sbs = dict()
        self._init_lrscoreplan()
        self._init_sbsscoreplan()

        # Initiate for subclass the scikit-learn pipeline
        self.model_sk = BaseEstimator()
        # Loop through the score plan
        self.model_lr = LrPruningConnector(
            scoreplan=self.scoreplan_lr,
            prefunc=None,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
        )
        self.cols_lrconnect = self.model_lr.outcols

        self.model_sbs = PipeSbsComparator(
            scoreplan=self.scoreplan_sbs,
            n_jobs=self.n_jobs
        )
        self.cols_sbs = self.model_sbs.outcols
        # Prepare the Pipe
        self.model_dp = DataPasser(on_cols=self.cols_lrconnect)

        # noinspection PyTypeChecker
        self.model_scorer = make_union(
            *[
                self.model_sbs,
                self.model_dp
            ]
        )
        self.cols_scorer = self.cols_lrconnect + self.cols_sbs
        ## Estimator
        self.imp = Imputer()
        self.model_sk = make_pipeline(
            *[
                self.model_scorer,
                self.imp,
                self.estimator
            ]
        )

        pass

    def _init_lrscoreplan(self):
        """
        Routine to initiate the class
        Dig into the code to understand.
        Args:

        Returns:
            None
        """
        self.scoreplan_lr = deepcopy(self.scoreplan)
        for inputfield in self.scoreplan:
            scoretype = self.scoreplan[inputfield]['type']
            # inputfield: 'name'
            # actualcolname 'name_ascii' or 'duns_cleaned'
            # If the field is not free text

            if scoretype == name_exact:
                # Rename column name with cleaned suffix
                actualcolname = '_'.join([inputfield, self._suffixid])
                self.scoreplan_lr[actualcolname] = self.scoreplan_lr.pop(inputfield)
            elif scoretype == name_freetext:
                # Rename column name with cleaned suffix
                actualcolname = '_'.join([inputfield, self._suffixascii])
                self.scoreplan_lr[actualcolname] = self.scoreplan_lr.pop(inputfield)
                # add the case on stop_words
                if self.scoreplan[inputfield].get(name_stop_words) is not None:
                    swcolname = actualcolname + '_' + self._suffixwosw
                    self.scoreplan_lr[swcolname] = self.scoreplan_lr[actualcolname].copy()
                    self.scoreplan_lr[swcolname][name_usescores] = ['ngram']
                    self.scoreplan_lr[swcolname][name_stop_words] = None
                    self.stop_words[actualcolname] = self.scoreplan[inputfield].get(name_stop_words)
        pass

    def _init_sbsscoreplan(self):
        self.scoreplan_sbs = deepcopy(self.scoreplan_lr)
        for actualcolname in self.scoreplan_lr:
            scoretype = self.scoreplan_lr[actualcolname]['type']
            # inputfield: 'name'
            # actualcolname 'name_ascii' or 'duns_cleaned'
            # If the field is not free text
            if scoretype == name_exact:
                # Remove exact ids already taken into account via the lridcomparator
                del self.scoreplan_sbs[actualcolname]
            elif scoretype == name_freetext:
                # DO NOTHING
                pass
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
                if actualcolname in self.stop_words.keys():
                    colnamewosw = '_'.join([actualcolname, self._suffixwosw])
                    new[colnamewosw] = new[actualcolname].apply(
                        lambda r: rmvstopwords(r, stop_words=self.stop_words[actualcolname])
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
        X_scores = self.model_lr.transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
        assert set(X_scores.columns.tolist()) == set(self.cols_lrconnect)
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
        X_scores = self.model_lr.transform(left=left, right=right, addvocab=addvocab, verbose=verbose)
        assert set(X_scores.columns.tolist()) == set(self.cols_lrconnect)
        X_sbs = createsbs(
            pairs=X_scores,
            left=left,
            right=right,
            use_cols=self.model_sbs.usecols,
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
            precision, recall = evalprecisionrecall(y_true=y_true, y_pred=X_sbs)
            print('{} | Pruning score: precision: {:.2%}, recall: {:.2%}'.format(pd.datetime.now(), precision, recall))
        # Redefine X_sbs and y_true to have a common index
        ix_common = y_true.index.intersection(
            X_sbs.index
        )
        y_true = y_true.loc[ix_common]
        X_sbs = X_sbs.loc[ix_common]

        # Fit the ScikitLearn Model
        self.model_sk.fit(X_sbs, y_true)

        if verbose:
            y_pred = pd.Series(self.model_sk.predict(X_sbs), index=X_sbs.index)
            scores = metrics(y_true=y_true, y_pred=y_pred)
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
            df_pred = self.model_sk.predict_proba(X_sbs)
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
                    names=self.ixnamepairs
                ),
                columns=[0, 1]
            )

        # add a case for missing lefts
        if addmissingleft is True:
            missing_lefts = newleft.index.difference(
                X_sbs.index.get_level_values(self.ixnameleft)
            )
            missing_lefts = pd.MultiIndex.from_product(
                [
                    missing_lefts,
                    [None]
                ],
                names=self.ixnamepairs
            )
            missing_lefts = pd.DataFrame(index=missing_lefts, columns=[0, 1])
            # Because those matches were not found: 100% probability of not being a match, 0% proba of being a match
            missing_lefts[0] = 1
            missing_lefts[1] = 0
            df_pred = pd.concat([df_pred, missing_lefts], axis=0, ignore_index=False)

        assert isinstance(df_pred, pd.DataFrame)
        assert set(df_pred.columns) == {0, 1}
        assert df_pred.index.names == self.ixnamepairs
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

    def evalpruning(self, left, right, y_true, addvocab='add', verbose=False):
        newleft = self._prepare_data(self.prefunc(left))
        newright = self._prepare_data(self.prefunc(right))
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true['y_true']

        X_sbs = self._lr_to_sbs(left=newleft, right=newright, addvocab=addvocab, verbose=verbose)
        precision, recall = evalprecisionrecall(y_true=y_true, y_pred=X_sbs)
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
        scores = self.scores(left=left, right=right, y_true=pairs)
        if kind != 'all':
            return scores[kind]
        else:
            return scores

    def scores(self, left, right, y_true):
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
        scores = metrics(y_true=y_true, y_pred=y_pred)
        return scores