import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from wookie.preutils import concatixnames, addsuffix


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
