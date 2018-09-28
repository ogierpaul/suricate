import numpy as np
import pandas as pd
from fuzzywuzzy.fuzz import ratio, token_set_ratio
from sklearn.base import TransformerMixin
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

        score = compare_token_series(
            train_series=train_series,
            new_series=new_series,
            tokenizer=self.tokenizer,
            new_ix=self.new_ix,
            train_ix=self.train_ix
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


def compare_token_series(new_series, train_series, tokenizer, new_ix='ix_new', train_ix='ix_train', fit=False):
    """
    return the cosine similarity of the tf-idf score of each possible pairs of documents
    Args:
        new_series (pd.Series): new data (To compare against train data)
        train_series (pd.Series): train data (To fit the tf-idf transformer)
        tokenizer: tokenizer of type sklearn
        new_ix (str): column name of new ix
        train_ix (str): column name of train ix
        fit (bool): if False, do not fit the tokenizer (is already fitted)
    Returns:
        pd.DataFrame
    """
    train_series = train_series.copy().dropna()
    new_series = new_series.copy().dropna()
    if fit is True:
        alldata = pd.concat([train_series, new_series], axis=0, ignore_index=True)
        tokenizer.fit(alldata)
    train_tfidf = tokenizer.transform(train_series.dropna())
    new_tfidf = tokenizer.transform(new_series)
    X = pd.DataFrame(cosine_similarity(new_tfidf, train_tfidf), columns=train_series.index)
    X[new_ix] = new_series.index
    score = pd.melt(
        X,
        id_vars=new_ix,
        var_name=train_ix,
        value_name='score'
    ).sort_values(by='score', ascending=False)
    return score


def tfidfpos(left, right, tokenizer, on, ixname='ix', fit=False, threshold=0):
    """

    Args:
        left (pd.DataFrame):
        right (pd.DataFrame):
        tokenizer:
        on (str):
        ixname (str):
        fit (bool):
        threshold (float):

    Returns:
        pd.DataFrame
    """
    score = compare_token_series(
        new_series=left[on],
        train_series=right[on],
        tokenizer=tokenizer,
        new_ix=ixname + '_left',
        train_ix=ixname + '_right',
        fit=fit
    )
    if not threshold is None:
        score = score.loc[score['score'] > threshold]
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
        s = (ratio(left, right) / 100)
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
        s = token_set_ratio(left, right) / 100
    return s


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


def preparevocab(col, df_left, df_right):
    """
    create a series with all the vocab wors
    Args:
        col (str): column to take for the vocabulary
        df_left (pd.DataFrame):
        df_right (pd.DataFrame)
    Returns:
        pd.Series
    """
    vocab = pd.Series(name=col)
    for df in [df_left, df_right]:
        vocab = pd.concat([vocab, df[col].dropna()], axis=0, ignore_index=True)
    return vocab


def innermatch(left, right, col, ixname='ix'):
    xleft = left[[col]].copy().dropna().reset_index(drop=False)
    xright = right[[col]].copy().dropna().reset_index(drop=False)
    x = pd.merge(left=xleft, right=xright, left_on=col, right_on=col, how='inner', suffixes=['_left', '_right'])
    x = x[[ixname + '_left', ixname + '_right']].set_index([ixname + '_left', ixname + '_right'])
    x[col + '_exact'] = 1
    return x


def idsmatch(left, right, ids, ixname='ix'):
    alldata = None
    for c in ids:
        pos = innermatch(left=left, right=right, col=c, ixname=ixname)
        if alldata is None:
            alldata = pos.copy()
        else:
            alldata = pd.merge(left=alldata, right=pos, left_index=True, right_index=True, how='outer')
    return alldata
