import pandas as pd
from sklearn.model_selection import train_test_split

from wookie import connectors, lrcomparators

if __name__ == '__main__':
    # Variable definition
    ## indexes
    ixname = 'ix'
    lsuffix = 'left'
    rsuffix = 'right'
    n_estimators = 5
    ixnameleft = ixname + '_' + lsuffix
    ixnameright = ixname + '_' + rsuffix
    ixnamepairs = [ixnameleft, ixnameright]
    ## File path
    filepath_left = '/Users/paulogier/81-GithubPackages/wookie/operations/data/left.csv'
    filepath_right = '/Users/paulogier/81-GithubPackages/wookie/operations/data/right.csv'
    filepath_training = '/Users/paulogier/81-GithubPackages/wookie/operations/data/trainingdata.csv'

    # Data Preparation
    # left = pd.read_csv(filepath_left, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    # right = pd.read_csv(filepath_right, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    df_train = pd.read_csv(filepath_training, dtype=str).set_index(ixnamepairs)
    df_train['y_true'] = df_train['y_true'].astype(float)
    df_train, df_test = train_test_split(df_train, train_size=0.7)
    train_left, train_right, y_train = connectors.separatesides(df_train)
    test_left, test_right, y_test = connectors.separatesides(df_test)
    con1 = lrcomparators.LrTokenComparator(
        on='name',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        scoresuffix='tfidf',
        vectorizermodel='tfidf',
        analyzer='word',
        ngram_range=(1, 1),
        strip_accents='ascii',
        store_threshold=0.2
    )
    # con2 = lrcomparators.LrIdComparator(
    #     on='duns',
    #     ixname=ixname,
    #     lsuffix=lsuffix,
    #     rsuffix=rsuffix
    # )
    # con3 = lrcomparators.LrTokenComparator(
    #     on='street',
    #     ixname=ixname,
    #     lsuffix=lsuffix,
    #     rsuffix=rsuffix,
    #     scoresuffix='tfidf',
    #     vectorizermodel='tfidf',
    #     analyzer='word',
    #     ngram_range=(1, 1),
    #     stop_words=['gmbh'],
    #     strip_accents='ascii',
    #     store_threshold=0.2
    # )
    conlist = [con1]
    model = lrcomparators.LrConnector(
        lrcomparators=conlist,
        pruning_thresholds={'name': 0.5, 'street': 0.6, 'duns': 1.0}
    )
    # X_scores = model.transform(left=train_left, right=train_right, verbose=True)
    # X_scores = model.transform(left=test_left, right=test_right, verbose=True)
    # # precision, recall = model._evalscore(left=train_left, right=train_right, y_true=y_train)
    # # # y_pred = con.transform(left=train_left, right=train_right, addvocab='add')
    # # # y_pred.loc[y_pred > 0] = 1
    for s, x, y_true, in zip(['train', 'test'], [[train_left, train_right], [test_left, test_right]],
                             [y_train, y_test]):
        print('{} | Starting pred on batch {}'.format(pd.datetime.now(), s))
        precision, recall = model._evalscore(left=x[0], right=x[1], y_true=y_true)
        print(
            '{} | Model score: precision: {:.2%}, recall: {:.2%}, on batch {}'.format(
                pd.datetime.now(),
                precision,
                recall,
                s
            )
        )
    pass
