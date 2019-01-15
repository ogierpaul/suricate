import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from wookie.preutils import concatixnames, rmvsuffix, addsuffix


def indexwithytrue(y_true, y_pred):
    """

    Args:
        y_true (pd.Series):
        y_pred (pd.Series):

    Returns:
        pd.Series: y_pred but with the missing indexes of y_true filled with 0
    """
    y_pred2 = pd.Series(index=y_true.index, name=y_pred.name)
    y_pred2.loc[y_true.index.intersection(y_pred.index)] = y_pred
    y_pred2.loc[y_true.index.difference(y_pred.index)] = 0
    return y_pred2


def _analyzeerrors(y_true, y_pred, rmvsameindex=True, ixnameleft='ix_left', ixnameright='ix_right'):
    ixnamepairs = [ixnameleft, ixnameright]
    y_true = y_true.copy()
    y_true.name = 'y_true'
    y_pred = y_pred.copy()
    y_pred.name = 'y_pred'
    pairs = pd.concat(
        [y_true, y_pred],
        axis=1
    )
    pairs['y_pred'].fillna(0, inplace=True)
    pairs = pairs.loc[
        ~(
            (pairs['y_pred'] == 0) & (
                pairs['y_true'] == 0
            )
        )
    ]
    pairs['correct'] = 'Ok'
    pairs.loc[
        (pairs['y_pred'] == 0) & (pairs['y_true'] == 1),
        'correct'
    ] = 'recall_error'
    pairs.loc[
        (pairs['y_pred'] == 1) & (pairs['y_true'] == 0),
        'correct'
    ] = 'precision_error'
    if rmvsameindex:
        pairs.reset_index(inplace=True)
        pairs = pairs[pairs[ixnameleft] != pairs[ixnameright]]
        pairs.set_index(ixnamepairs, inplace=True)
    return pairs


def metrics(y_true, y_pred):
    y_pred2 = indexwithytrue(y_true=y_true, y_pred=y_pred)
    scores = dict()
    scores['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred2)
    scores['precision'] = precision_score(y_true=y_true, y_pred=y_pred2)
    scores['recall'] = recall_score(y_true=y_true, y_pred=y_pred2)
    scores['f1'] = f1_score(y_true=y_true, y_pred=y_pred2)
    return scores


def evalprecisionrecall(y_true, y_pred):
    """

    Args:
        y_pred (pd.DataFrame/pd.Series): everything that is index is counted as true
        y_true (pd.Series):

    Returns:
        float, float: precision and recall
    """
    true_pos = y_true.loc[y_true > 0]
    true_neg = y_true.loc[y_true == 0]
    # EVERYTHING THAT IS CAUGHT BY Y_PRED IS CONSIDERED AS TRUE
    catched_pos = y_pred.loc[true_pos.index.intersection(y_pred.index)]
    catched_neg = y_pred.loc[y_pred.index.difference(catched_pos.index)]
    missed_pos = true_pos.loc[true_pos.index.difference(y_pred.index)]
    assert true_pos.shape[0] + true_neg.shape[0] == y_true.shape[0]
    assert catched_pos.shape[0] + catched_neg.shape[0] == y_pred.shape[0]
    assert catched_pos.shape[0] + missed_pos.shape[0] == true_pos.shape[0]
    recall = catched_pos.shape[0] / true_pos.shape[0]
    precision = catched_pos.shape[0] / y_pred.shape[0]
    return precision, recall


def evalpred(y_true, y_pred, verbose=True, namesplit=None):
    if namesplit is None:
        sset = ''
    else:
        sset = 'for set {}'.format(namesplit)
    precision, recall = evalprecisionrecall(y_true=y_true, y_pred=y_pred)
    if verbose:
        print(
            '{} | Pruning score: precision: {:.2%}, recall: {:.2%} {}'.format(
                pd.datetime.now(), precision, recall, sset
            )
        )
    scores = metrics(y_true=y_true, y_pred=y_pred)
    if verbose:
        print(
            '{} | Model score: precision: {:.2%}, recall: {:.2%} {}'.format(
                pd.datetime.now(), scores['precision'], scores['recall'], sset
            )
        )
    return scores


def cartesian_join(left, right, lsuffix='left', rsuffix='right'):
    """

    Args:
        left (pd.DataFrame): table 1
        right (pd.DataFrame): table 2
        lsuffix (str):
        rsuffix (str):

    Returns:
        pd.DataFrame

    Examples:
        df1 = pd.DataFrame({'a':['foo', 'bar']})
             a
        0   foo
        1	bar

        df2 = pd.DataFrame({'b':['foz', 'baz']})
             b
        0   foz
        1	baz

        cartesian_join(df1, df2)
            index_left	a_left	index_right	b_right
        0	0	        foo	    0	        foz
        1	0	        foo	    1	        baz
        2	1	        bar	    0	        foz
        3	1	        bar	    1	        baz

    """

    def rename_with_suffix(df, suffix):
        """
        rename the columns with a suffix, including the index
        Args:
            df (pd.DataFrame): {'ix':['name']}
            suffix (str): 'left'

        Returns:
            pd.DataFrame

        Examples:
            df = pd.DataFrame({'a':['foo', 'bar']})
                 a
            0   foo
            1	bar

            rename_with_suffix(df, 'right')

                index_right	a_right
            0	0	        foo
            1	1	        bar
        """
        if suffix is None:
            return df
        assert isinstance(suffix, str)
        assert isinstance(df, pd.DataFrame)
        df_new = df.copy()
        if df.index.name is None:
            ixname = 'ix'
        else:
            ixname = df.index.name
        df_new.index.name = ixname
        df_new.reset_index(drop=False, inplace=True)
        cols = df_new.columns
        mydict = dict(
            zip(
                cols,
                map(lambda c: c + '_' + suffix, cols)
            )
        )
        df_new.rename(columns=mydict, inplace=True)
        return df_new

    # hack to create a column name unknown to both df1 and df2
    tempcolname = 'f1b3'
    while tempcolname in left.columns or tempcolname in right.columns:
        tempcolname += 'f'

    # create a new df1 with renamed cols
    df1new = rename_with_suffix(left, lsuffix)
    df2new = rename_with_suffix(right, rsuffix)
    df1new[tempcolname] = 0
    df2new[tempcolname] = 0
    dfnew = pd.merge(df1new, df2new, on=tempcolname).drop([tempcolname], axis=1)
    del df1new, df2new, tempcolname

    return dfnew


def separatesides(df, ixname='ix', lsuffix='left', rsuffix='right', y_true_col='y_true'):
    """
    Separate a side by side training table into the left table, the right table, and the list of pairs
    Args:
        df (pd.DataFrame): side by side dataframe {['ix_left', 'ix_right'] :['name_left', 'name_right']}
        lsuffix (str): left suffix 'left'
        rsuffix (str): right suffix 'right'
        y_true_col (str): name of y_true column
        ixname (str): name in index column

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series : {ix:['name'}, {'ix':['name'} {['ix_left', 'ix_right']:y_true}
    """
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )

    def takeside(df, suffix, ixname):
        """

        Args:
            df (pd.DataFrame):
            suffix (str):
            ixname (str):

        Returns:

        """
        new = df.copy().reset_index(drop=False)
        new = new[list(filter(lambda r: r[-len(suffix):] == suffix, new.columns))]
        new = rmvsuffix(new, suffix).drop_duplicates(subset=[ixname])
        new.set_index([ixname], inplace=True)
        return new

    xleft = takeside(df=df, suffix=lsuffix, ixname=ixname)
    xright = takeside(df=df, suffix=rsuffix, ixname=ixname)
    pairs = df.loc[:, y_true_col].copy()
    return xleft, xright, pairs


def createsbs(pairs, left, right, use_cols=None, ixname='ix', lsuffix='left', rsuffix='right'):
    """
    Create a side by side table from a list of pairs (as a DataFrame)
    Args:
        pairs (pd.DataFrame/pd.Series): of the form {['ix_left', 'ix_right']:['y_true']}
        left (pd.DataFrame): of the form ['name'], index=ixname
        right (pd.DataFrame): of the form ['name'], index=ixname
        use_cols (list): columns to use
        lsuffix (str): default 'left'
        rsuffix (str): default 'right'
        ixname (str): default 'ix' name of the index

    Returns:
        pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
    """
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )
    if use_cols is None or len(use_cols) == 0:
        use_cols = left.columns.intersection(right.columns)
    xleft = left[use_cols].copy().reset_index(drop=False)
    xright = right[use_cols].copy().reset_index(drop=False)
    xpairs = pairs.copy().reset_index(drop=False)
    if isinstance(xpairs, pd.Series):
        xpairs = pd.DataFrame(xpairs)
    xleft = addsuffix(xleft, lsuffix).set_index(ixnameleft)
    xright = addsuffix(xright, rsuffix).set_index(ixnameright)
    sbs = xpairs.join(
        xleft, on=ixnameleft, how='left'
    ).join(
        xright, on=ixnameright, how='left'
    ).set_index(
        ixnamepairs
    )
    return sbs


def showpairs(pairs, left, right, use_cols=None, ixname='ix', lsuffix='left', rsuffix='right', filterzeroes=False):
    """
    Like createsbs, but reorder the columns to compare left and right columns
    Args:
        pairs (pd.DataFrame/pd.Series): {[ix_left, ix_right]: col}
        left (pd.DataFrame): {ix: [cols]}
        right (pd.DataFrame): {ix: [cols]}
        use_cols (list): [name, duns, ..]
        ixname (str):
        lsuffix (str):
        rsuffix (str):
        filterzeroes (bool):

    Returns:
        pd.DataFrame: {[ix_left, ix_right]: [name_left, name_right, duns_left, duns_right]}
    """
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix
    )
    if isinstance(pairs, pd.Series):
        xpairs = pd.DataFrame(pairs).copy()
    else:
        xpairs = pairs.copy()
    if use_cols is None:
        use_cols = left.columns.intersection(right.columns)
    res = createsbs(pairs=xpairs, left=left, right=right, use_cols=use_cols,
                    ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
    displaycols = xpairs.columns.tolist()
    for c in use_cols:
        displaycols.append(c + '_' + lsuffix)
        displaycols.append(c + '_' + rsuffix)
    res = res[displaycols]
    return res


class DFConnector:
    def __init__(self, left, right, on=None, ixname='ix',
                 lsuffix='left', rsuffix='right', scoresuffix='score',
                 n_jobs=1, store_ths=1.0):
        """

        Args:
            left (pd.DataFrame):
            right (pd.DataFrame):
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            on (str): name of the column on which to do the join
            scoresuffix (str): name of the score suffix
            n_jobs (int):
            store_ths (float):
        """
        assert isinstance(left, pd.DataFrame)
        assert isinstance(right, pd.DataFrame)
        self.leftdf = left
        self.rightdf = right
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix
        )
        self.on = on
        self.scoresuffix = scoresuffix
        self.outcol = self.on + '_' + self.scoresuffix
        self.n_jobs = n_jobs
        self.store_ths = store_ths
        pass

    def showpairs(self, pairs, use_cols=None):
        """
        Create a side by side table from a list of pairs (as a DataFrame)
        Args:
            pairs (pd.DataFrame/pd.Series): of the form {['ix_left', 'ix_right']:['y_true']}
            use_cols (list): columns to use

        Returns:
            pd.DataFrame {['ix_left', 'ix_right'] : ['name_left', 'name_right', .....]}
        """
        if use_cols is None or len(use_cols) == 0:
            use_cols = self.leftdf.columns.intersection(self.rightdf.columns)
        xleft = self.leftdf[use_cols].copy().reset_index(drop=False)
        xright = self.rightdf[use_cols].copy().reset_index(drop=False)
        xpairs = pairs.copy().reset_index(drop=False)
        if isinstance(xpairs, pd.Series):
            xpairs = pd.DataFrame(xpairs)

        xleft = addsuffix(xleft, self.lsuffix).set_index(self.ixnameleft)
        xright = addsuffix(xright, self.rsuffix).set_index(self.ixnameright)
        sbs = xpairs.join(
            xleft, on=self.ixnameleft, how='left'
        ).join(
            xright, on=self.ixnameright, how='left'
        ).set_index(
            self.ixnamepairs
        )
        return sbs

    def fit(self):
        return self

    def transform(self):
        """

        Returns:
            pd.Series
        """
        return pd.Series()

    def fit_transform(self):
        self.fit()
        return self.transform()

    def sidebyside(self):
        """

        Returns:
            pd.DataFrame
        """
        y_score = self.transform()
        sbs = self.showpairs(pairs=y_score)
        return sbs

    def pruning_score(self, y_true):
        """
        compression: defined by the number of possible pairs divided by the number of actual pairs
        precision and recall : depends on y_true
        Args:
            y_true: list of pairs in the index

        Returns:
            dict: ['compression', 'precision', 'recall']
        """
        score = dict()
        sbs = self.transform()
        score['compression'] = (self.leftdf.shape[0] * self.rightdf.shape[0]) / sbs.shape[0]
        precision, recall = evalprecisionrecall(y_true=y_true, y_pred=sbs)
        score['precision'] = precision
        score['recall'] = recall
        return score

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
