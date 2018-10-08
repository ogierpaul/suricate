import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    X.fillna(0, inplace=True)
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
