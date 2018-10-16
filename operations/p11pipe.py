import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as Clf
from sklearn.model_selection import train_test_split

import wookie
from operations import companypreparation as preprocessing

if __name__ == '__main__':
    # Variable definition
    ## indexes
    ixname = 'ix'
    lsuffix = '_left'
    rsuffix = '_right'
    n_estimators = 500
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
    train_left, train_right, y_train = wookie.separatesides(df_train)
    test_left, test_right, y_test = wookie.separatesides(df_test)
    dedupe = wookie.LrDuplicateFinder(
        prefunc=preprocessing.preparedf,
        scoreplan={
            'name': {
                'type': 'FreeText',
                'stop_words': preprocessing.companystopwords,
                'use_scores': ['tfidf', 'ngram'],
                'threshold': 0.6,
            },
            'street': {
                'type': 'FreeText',
                'stop_words': preprocessing.streetstopwords,
                'use_scores': ['tfidf', 'ngram', 'token'],
                'threshold': 0.6
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
        verbose=False
    )
    dedupe.fit(
        left=train_left,
        right=train_right,
        y_true=y_train,
        verbose=False
    )

    for s, x, y_true, in zip(['train', 'test'], [[train_left, train_right], [test_left, test_right]],
                             [y_train, y_test]):
        print('\n****************\n')
        print('{} | Starting pred on batch {}'.format(pd.datetime.now(), s))
        precision, recall = dedupe.evalpruning(left=x[0], right=x[1], y_true=y_true, verbose=False)
        print(
            '{} | Pruning score: precision: {:.2%}, recall: {:.2%}, on batch {}'.format(
                pd.datetime.now(),
                precision,
                recall,
                s
            )
        )
        scores = dedupe.scores(left=x[0], right=x[1], y_true=y_true)
        print(
            '{} | Model score: precision: {:.2%}, recall: {:.2%}, on batch {}'.format(
                pd.datetime.now(),
                scores['precision'],
                scores['recall'],
                s
            )
        )
    pass