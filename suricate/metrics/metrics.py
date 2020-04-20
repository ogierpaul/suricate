from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score


def get_commonscores(y_true, y_pred):
    """
    Calculate the precision, recall, f1, accuracy, balanced_accuracy scores
    Args:
        y_true (pd.Series): y_true (with limited set of index)
        y_pred (pd.Series): y_pred (with limited set of index)

    Returns:
        dict : scores calculated on intersection of index. Keys Precision, recall, f1, accuracy, balanced_accuracy
    """
    commonindex = y_true.index.intersection(y_pred.index)
    myscores = dict()
    y2_true = y_true.loc[commonindex]
    y2_pred = y_pred.loc[commonindex]
    myscores['precision'] = precision_score(y_true=y2_true, y_pred=y2_pred)
    myscores['recall'] = recall_score(y_true=y2_true, y_pred=y2_pred)
    myscores['f1'] = f1_score(y_true=y2_true, y_pred=y2_pred)
    myscores['accuracy'] = accuracy_score(y_true=y2_true, y_pred=y2_pred)
    myscores['balanced_accuracy'] = balanced_accuracy_score(y_true=y2_true, y_pred=y2_pred)
    return myscores


suffixexact = 'exact'
suffixtoken = 'token'
suffixfuzzy = 'simple'
name_freetext = 'FreeText'
name_exact = 'Exact'
name_pruning_threshold = 'threshold'
name_usescores = 'use_scores'
name_stop_words = 'stop_words'
navalue_score = 0