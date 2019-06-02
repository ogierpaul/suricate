import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wookie.lrdftransformers.base import LrDfTransformerMixin
from wookie.preutils import concatixnames


# HERE I HAVE VERSION 2 in try to optimize performance

class VectorizerConnector(LrDfTransformerMixin):
    def __init__(self,
                 on,
                 ixname='ix',
                 lsuffix='left',
                 rsuffix='right',
                 scoresuffix='vec',
                 vecmodel='tfidf',
                 ngram_range=(1, 2),
                 stop_words=None,
                 strip_accents='ascii',
                 analyzer='word',
                 addvocab='add',
                 **kwargs):
        """

        Args:
            on (str): name of column on which to do the pivot
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): name of score suffix to be added at the end
            pruning_ths (float): threshold to keep
            vecmodel (str): ['tfidf', 'cv']: use either Sklearn TfIdf Vectorizer or Count Vectorizer
            ngram_range (tuple): default (1,1)
            stop_words (list) : {'english'}, list, or None (default)
            strip_accents (str): {'ascii', 'unicode', None}
            analyzer (str): {'word', 'char'} or callable. Whether the feature should be made of word or chars n-grams.
            n_jobs (int): number of jobs
            analyzer (str): ['word', 'char']
            addvocab (str): in ['add', 'keep', 'replace']
                'add' --> add new vocabulary to current
                'keep' --> only keep current vocabulary
                'replace' --> replace current vocabulary with new
            pruning (bool)
        """

        LrDfTransformerMixin.__init__(self,
                                      ixname=ixname,
                                      lsuffix=lsuffix,
                                      rsuffix=rsuffix,
                                      on=on,
                                      scoresuffix=scoresuffix + '_' + analyzer,
                                      **kwargs)
        self._vocab = list()
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.strip_accents = strip_accents
        self.vecmodel = vecmodel
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.strip_accents = self.strip_accents
        self.addvocab = addvocab
        if vecmodel == 'tfidf':
            vectorizerclass = TfidfVectorizer
        elif vecmodel == 'cv':
            vectorizerclass = CountVectorizer
        else:
            raise ValueError('{} not in [cv, tfidf]'.format(vecmodel))
        self.vectorizer = vectorizerclass(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            strip_accents=self.strip_accents
        )
        pass

    def _fit(self, X=None, y=None):
        return self

    def _transform(self, X, on_ix=None):
        """
        Add, keep, or replace new vocabulary to the vectorizer
        Fit the tokenizer with the new vocabulary
        Calculate the cosine_simlarity score for the left and right columns \
            using the output of the vectorizer
        Args:
            X (list)
        Returns:
            pd.Series : {['ix_left', 'ix_right']: 'name_tfidf'}
        """
        ix = self._getindex(X=X, y=on_ix)
        newleft, newright = self._toseries(left=X[0], right=X[1], on_ix=ix)
        # Fit
        if self.addvocab in ['add', 'replace']:
            self._vocab = _update_vocab(left=newleft, right=newright, vocab=self._vocab, addvocab=self.addvocab)
            self.vectorizer = self.vectorizer.fit(self._vocab)
        score = _transform_tkscore(
            left=newleft,
            right=newright,
            vectorizer=self.vectorizer,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            outcol=self.outcol
        )
        score.name = self.outcol
        return score


class _VectorizerConnector2_temp(LrDfTransformerMixin):
    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right',
                 scoresuffix='vec', vecmodel='tfidf', ngram_range=(1, 2), n_jobs=1,
                 stop_words=None, strip_accents='ascii', analyzer='word', addvocab='add', **kwargs):
        """

        Args:
            on (str): name of column on which to do the pivot
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): name of score suffix to be added at the end
            pruning_ths (float): threshold to keep
            vecmodel (str): ['tfidf', 'cv']: use either Sklearn TfIdf Vectorizer or Count Vectorizer
            ngram_range (tuple): default (1,1)
            stop_words (list) : {'english'}, list, or None (default)
            strip_accents (str): {'ascii', 'unicode', None}
            analyzer (str): {'word', 'char'} or callable. Whether the feature should be made of word or chars n-grams.
            n_jobs (int): number of jobs
            analyzer (str): ['word', 'char']
            addvocab (str): in ['add', 'keep', 'replace']
                'add' --> add new vocabulary to current
                'keep' --> only keep current vocabulary
                'replace' --> replace current vocabulary with new
            pruning (bool)
        """

        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      n_jobs=n_jobs, scoresuffix=scoresuffix + '_' + analyzer,
                                      **kwargs)
        self._vocab = list()
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.strip_accents = strip_accents
        self.vecmodel = vecmodel
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.strip_accents = self.strip_accents
        self.addvocab = addvocab
        if vecmodel == 'tfidf':
            vectorizerclass = TfidfVectorizer
        elif vecmodel == 'cv':
            vectorizerclass = CountVectorizer
        else:
            raise ValueError('{} not in [cv, tfidf]'.format(vecmodel))
        self.vectorizer = vectorizerclass(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            strip_accents=self.strip_accents
        )
        pass

    def _fit(self, X=None, y=None):
        self._vocab = _update_vocab(
            left=X[0][self.on],
            right=X[1][self.on],
            vocab=None,
            addvocab='replace'
        )
        self.vectorizer.fit(self._vocab)
        return self

    def _transform(self, X, on_ix=None):
        """
        Add, keep, or replace new vocabulary to the vectorizer
        Fit the tokenizer with the new vocabulary
        Calculate the cosine_simlarity score for the left and right columns \
            using the output of the vectorizer
        Args:
            X (list)
        Returns:
            pd.Series : {['ix_left', 'ix_right']: 'name_tfidf'}
        """
        ix = self._getindex(X=X, y=on_ix)
        # newleft, newright = self._toseries(left=X[0], right=X[1], on_ix = ix)
        # Fit
        score = _transform_tkscore2_temp(
            left=X[0][self.on],
            right=X[1][self.on],
            vectorizer=self.vectorizer,
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            outcol=self.outcol
        )
        score.name = self.outcol
        return score


def _update_vocab(left, right, vocab=None, addvocab='add'):
    """
    Update the list of existing vocabulary for a column
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

    Examples:

    """

    def _check_vocab(vocab):
        if vocab is None:
            return list()
        else:
            assert isinstance(vocab, list)
            return vocab

    assert isinstance(left, pd.Series)
    assert isinstance(right, pd.Series)
    assert addvocab in ['add', 'keep', 'replace']
    if addvocab == 'keep':
        return _check_vocab(vocab)
    else:
        new = (left.dropna().tolist() + right.dropna().tolist()).copy()
        if addvocab == 'replace':
            return new
        else:
            if addvocab == 'add':
                return _check_vocab(vocab) + new
            elif addvocab == 'replace':
                return new


def _transform_tkscore2_temp(left,
                             right,
                             vectorizer,
                             ixname='ix',
                             lsuffix='left',
                             rsuffix='right',
                             outcol='score'):
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
    assert isinstance(left, pd.Series)
    assert isinstance(right, pd.Series)
    tkr = vectorizer.transform(np.where(pd.isnull(right.values), '', right.values))
    tkl = vectorizer.transform(np.where(pd.isnull(left.values), '', left.values))
    score = cosine_similarity(tkl, tkr).reshape(-1, 1).flatten()

    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )
    y = pd.Series(index=
    # TODO: write index)
    # If we cannot find a single value we return blank
    if left.shape[0] == 0 or right.shape[0] == 0:
        ix =
        r = pd.Series(index=ix, name=scorename)
        return r

    return score[scorename]


def _transform_tkscore(left,
                       right,
                       vectorizer,
                       ixname='ix',
                       lsuffix='left',
                       rsuffix='right',
                       outcol='score'):
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
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )
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
    score = score[scorename]
    return score
