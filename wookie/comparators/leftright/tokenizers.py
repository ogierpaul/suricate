import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wookie.base import BaseLrComparator
from wookie.preutils import concatixnames

_tfidf_store_threshold_value = 0.5

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
    if threshold is not None:
        score = score[score[scorename] >= threshold]
    score = score[scorename]
    return score

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
