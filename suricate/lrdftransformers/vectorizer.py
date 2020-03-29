import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from suricate.lrdftransformers.base import LrDfTransformerMixin


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

    def _transform(self, X):
        """
        Add, keep, or replace new vocabulary to the vectorizer
        Fit the tokenizer with the new vocabulary
        Calculate the cosine_simlarity score for the left and right columns \
            using the output of the vectorizer
        Args:
            X (list)
        Returns:
            np.ndarray:  of shape(n_samples_left * n_samples_right, 1)
        """
        ix = self._getindex(X=X)
        newleft, newright = self._toseries(left=X[0], right=X[1], on_ix=ix)
        # Fit
        if self.addvocab in ['add', 'replace']:
            self._vocab = _update_vocab(left=newleft, right=newright, vocab=self._vocab, addvocab=self.addvocab)
            self.vectorizer = self.vectorizer.fit(self._vocab)
        score = _transform_tkscore(
            left=X[0][self.on],
            right=X[1][self.on],
            vectorizer=self.vectorizer
        )
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


def _transform_tkscore(left,
                       right,
                       vectorizer):
    """
    Args:
        left (pd.Series):
        right (pd.Series):
        vectorizer: TfIdf vectorizer or CountVectorizer
    Returns:
        pd.Series
    """
    assert isinstance(left, pd.Series)
    assert isinstance(right, pd.Series)
    tkl = vectorizer._transform(_fillwithblanks(left.values))
    tkr = vectorizer._transform(_fillwithblanks(right.values))
    score = np.nan_to_num(cosine_similarity(tkl, tkr)).reshape(-1, 1)
    return score


def _fillwithblanks(y, replace_with=''):
    """

    Args:
        y (np.ndarray):
        replace_with (str): string to replace with

    Returns:
        np.ndarray with Null values replaced by ''
    """
    return np.where(pd.isnull(y), replace_with, y).flatten()
