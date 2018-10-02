import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio as simpleratio, partial_token_set_ratio as tokenratio
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_union


class BaseComparator(TransformerMixin):
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

    def fit(self, X, y=None):
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
            assert all(map(lambda c: c in X.columns, self.on_cols))
            res = X[self.on_cols]
        else:
            res = X
        return res


class FuzzyWuzzyComparator(BaseComparator, TransformerMixin):
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
            compfunc = exact_score
        elif comparator == 'fuzzy':
            compfunc = fuzzy_score
        elif comparator == 'token':
            compfunc = token_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token']")
        BaseComparator.__init__(
            self,
            compfunc=compfunc,
            left=left,
            right=right,
            *args,
            **kwargs
        )
        pass


class PipeComparator(TransformerMixin):
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
                    FuzzyWuzzyComparator(left=left, right=right, comparator=v)
                )
        pipe = make_union(n_jobs=n_jobs, *stages, *args, **kwargs)
        res = pipe.fit_transform(X)

        return res


class TokenComparator(TransformerMixin):
    def __init__(self, tokenizer, new_ix='new_ix', train_ix='train_ix', new_col='name', train_col='name'):
        """

        Args:
            tokenizer (TfidfVectorizer): Tokenizer
        """
        TransformerMixin.__init__(self)
        self.tokenizer = tokenizer
        self.new_ix = new_ix
        self.train_ix = train_ix
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


class Tfidf_Connector(TransformerMixin):
    def __init__(self, on, left, right, ixname='ix', lsuffix='_left', rsuffix='_right', threshold=0, **kwargs):
        TransformerMixin.__init__(self)
        self.on = on
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.vocab = pd.Series()
        self.left = left
        self.right = right
        self.ix_left = ixname + lsuffix
        self.ix_right = ixname + rsuffix
        self.ixs = [self.ix_left, self.ix_right]
        self.scorename = self.on + '_tfidf'
        self.threshold = threshold
        self.tokenizer = TfidfVectorizer(**kwargs)
        pass

    def fit(self, X=None, y=None, addvocab=True):
        left = self.left[self.on].dropna().copy()
        right = self.right[self.on].dropna().copy()
        assert isinstance(left, pd.Series)
        assert isinstance(right, pd.Series)
        vocab2 = pd.concat(
            [left, right],
            axis=0,
            ignore_index=True
        )
        if addvocab is True:
            self.vocab = pd.concat(
                [self.vocab, addvocab],
                axis=0,
                ignore_index=True
            )
        else:
            self.vocab = vocab2
        self.tokenizer.fit(self.vocab)
        return self

    def transform(self, X=None):
        left = self.left[self.on].dropna().copy()
        right = self.right[self.on].dropna().copy()
        right_tfidf = self.tokenizer.transform(left)
        left_tfidf = self.tokenizer.transform(right)
        X = pd.DataFrame(cosine_similarity(left_tfidf, right_tfidf), columns=right)
        X[self.ix_left] = left.index
        score = pd.melt(
            X,
            id_vars=self.ix_left,
            var_name=self.ix_right,
            value_name=self.scorename
        ).set_index(
            self.ixs
        ).sort_values(
            by=self.scorename,
            ascending=False
        )
        if not self.threshold is None:
            score = score.loc[score[self.scorename] > self.threshold]
        return score

    def update(self, left, right):
        self.left = left
        self.right = right
        pass


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
    right_tfidf = tokenizer.transform(right.dropna())
    left_tfidf = tokenizer.transform(left)
    X = pd.DataFrame(cosine_similarity(left_tfidf, right_tfidf), columns=right.index)
    X[ix_left] = left.index
    score = pd.melt(
        X,
        id_vars=ix_left,
        var_name=ix_right,
        value_name='score'
    ).sort_values(by='score', ascending=False)
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


navalue_score = None


def valid_inputs(left, right):
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


def exact_score(left, right):
    """
    Checks if the two values are equali
    Args:
        left (object): object number 1
        right (object): object number 2

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        return float(left == right)


def fuzzy_score(left, right):
    """
    return ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        s = (simpleratio(left, right) / 100)
    return s


def token_score(left, right):
    """
    return the token_set_ratio score of fuzzywuzzy
    Args:
        left (str): string number 1
        right (str): string number 2

    Returns:
        float
    """
    if valid_inputs(left, right) is False:
        return navalue_score
    else:
        s = tokenratio(left, right) / 100
    return s


def preparevocab(on, left, right):
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
