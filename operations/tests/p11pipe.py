import pandas as pd
from sklearn.model_selection import train_test_split

from operations import companypreparation as preprocessing
from wookie import connectors
from wookie.lrcomparators import LrDuplicateFinder
from wookie.preutils import concatixnames

if __name__ == '__main__':
    # Variable definition
    ## indexes
    ixname = 'ix'
    lsuffix = 'left'
    rsuffix = 'right'
    ixnameleft, ixnameright, ixnamepairs = concatixnames(
        ixname=ixname,
        lsuffix=lsuffix,
        rsuffix=rsuffix
    )
    n_estimators = 5
    dem_treshold = 0.5
    nrows = None
    scoreplan_origin = {
        'name': {
            'type': 'FreeText',
            'stop_words': preprocessing.companystopwords,
            'threshold': dem_treshold
        },
        'street': {
            'type': 'FreeText',
            'stop_words': preprocessing.streetstopwords,
            'threshold': dem_treshold,
        },
        'city': {
            'type': 'FreeText',
            'stop_words': preprocessing.citystopwords,
            'threshold': None,
        },
        'duns': {
            'type': 'Exact',
            'threshold': 1.0
        },
        'postalcode': {
            'type': 'FreeText',
            'threshold': None,
            'use_scores': ['fuzzy']
        },
        'countrycode': {
            'type': 'Exact',
            'threshold': None
        }
    }

    ## File path
    filepath_left = '/Users/paulogier/81-GithubPackages/wookie/operations/data/left.csv'
    filepath_right = '/Users/paulogier/81-GithubPackages/wookie/operations/data/right.csv'
    filepath_training = '/Users/paulogier/81-GithubPackages/wookie/operations/data/trainingdata.csv'

    # Data Preparation
    # left = pd.read_csv(filepath_left, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    # right = pd.read_csv(filepath_right, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    df_train = pd.read_csv(filepath_training, nrows=nrows).set_index(ixnamepairs)
    df_train, df_test = train_test_split(df_train, train_size=0.7)
    train_left, train_right, y_train = connectors.separatesides(df_train)
    test_left, test_right, y_test = connectors.separatesides(df_test)

    dedupe = LrDuplicateFinder(
        scoreplan=scoreplan_origin,
        prefunc=preprocessing.preparedf,
        verbose=True,
        n_jobs=4
    )

    dedupe.fit(
        left=train_left,
        right=train_right,
        y_true=y_train,
        verbose=True
    )
    print('\n', sorted(dedupe.cols_scorer), '\n')

    for s, x, y_true, in zip(['train', 'tests'], [[train_left, train_right], [test_left, test_right]],
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
