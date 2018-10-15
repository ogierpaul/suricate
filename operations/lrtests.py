import pandas as pd

from wookie import connectors, lrcomparators

if __name__ == '__main__':
    # Variable definition
    ## indexes
    ixname = 'ix'
    lsuffix = '_left'
    rsuffix = '_right'
    n_estimators = 5
    ixnameleft = ixname + lsuffix
    ixnameright = ixname + rsuffix
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
    # df_train, df_test = train_test_split(df_train, train_size=0.7)
    train_left, train_right, y_train = connectors.separatesides(df_train)
    # test_left, test_right, y_test = connectors.separatesides(df_test)
    con = lrcomparators.LrTokenComparator(
        on='street',
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        scoresuffix='tfidf',
        vectorizermodel='tfidf',
        analyzer='word',
        ngram_range=(1, 1),
        stop_words=['gmbh'],
        strip_accents='ascii',
        store_threshold=0.2
    )
    con = lrcomparators.LrIdComparator(
        on='duns'
    )
    con.fit(left=train_left, right=train_right)
    precision, recall = con._evalscore(left=train_left, right=train_right, y_true=y_train)
    # y_pred = con.transform(left=train_left, right=train_right, addvocab='add')
    # y_pred.loc[y_pred > 0] = 1
    # precision, recall = comparators._evalprecisionrecall(y_true=y_train, y_pred=y_pred)
    print(
        '{} | Model score: precision: {:.2%}, recall: {:.2%}'.format(
            pd.datetime.now(),
            precision,
            recall
        )
    )
    pass