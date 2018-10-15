import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as Clf
from sklearn.model_selection import train_test_split

import wookie.lrcomparators
from operations import companypreparation as preprocessing
from wookie import connectors, comparators

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
    df_train = pd.read_csv(filepath_training).set_index(ixnamepairs)
    df_train, df_test = train_test_split(df_train, train_size=0.7)
    train_left, train_right, y_train = connectors.separatesides(df_train)
    test_left, test_right, y_test = connectors.separatesides(df_test)
    dedupe = wookie.lrcomparators.LrDuplicateFinder(
        prefunc=preprocessing.preparedf,
        scoreplan={
            'name': {
                'type': 'FreeText',
                'stop_words': preprocessing.companystopwords,
                'use_scores': ['tfidf', 'ngram', 'fuzzy', 'token'],
                'threshold': 0.4,
            },
            'street': {
                'type': 'FreeText',
                'stop_words': preprocessing.streetstopwords,
                'use_scores': ['tfidf', 'ngram', 'fuzzy', 'token'],
                'threshold': 0.4
            },
            'city': {
                'type': 'FreeText',
                'stop_words': preprocessing.citystopwords,
                'use_scores': ['tfidf', 'ngram', 'fuzzy'],
                'threshold': None
            },
            'duns': {'type': 'Id'},
            'postalcode': {'type': 'Code'},
            'countrycode': {'type': 'Category'}
        },
        estimator=Clf(n_estimators=n_estimators),
        verbose=True
    )
    dedupe.fit(
        left=train_left,
        right=train_right,
        pairs=y_train,
        verbose=True
    )
    for s, y_true, in zip(['train', 'test'], [y_train, y_test]):
        print('{} | Starting pred on batch {}'.format(pd.datetime.now(), s))
        y_pred = dedupe.predict(left=train_left, right=train_right)
        comparators._evalpred(y_true=y_true, y_pred=y_pred, namesplit=s)
    # singlegrouper = grouping.SingleGrouping(dedupe=dedupe)
    # singlegrouper.findduplicates(data=train_right, n_batches=None, n_records=30)
    # singlegrouper.data.to_csv('results_right.csv')
    pass