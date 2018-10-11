import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as Clf

from operations import companypreparation as preprocessing
from wookie import connectors, comparators, grouping

if __name__ == '__main__':
    # Variable definition
    ## indexes
    ixname = 'ix'
    lsuffix = '_left'
    rsuffix = '_right'
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
    # df_train, df_test = train_test_split(data, train_size=0.5)
    train_left, train_right, y_train = connectors.separatesides(df_train)
    # test_left, test_right, y_test = connectors.separatesides(df_test)
    dedupe = comparators.LrDuplicateFinder(
        prefunc=preprocessing.preparedf,
        scoreplan={
            'name': {
                'type': 'FreeText',
                'stop_words': preprocessing.companystopwords,
                'threshold': 0.3,
            },
            'street': {
                'type': 'FreeText',
                'stop_words': preprocessing.streetstopwords,
                'threshold': 0.3
            },
            'city': {
                'type': 'FreeText',
                'stop_words': preprocessing.citystopwords,
                'threshold': None
            },
            'duns': {'type': 'Id'},
            'postalcode': {'type': 'Code'},
            'countrycode': {'type': 'Category'},
            'postalcode_2char': {'type': 'Code'}
        },
        estimator=Clf(n_estimators=1000),
        verbose=True
    )
    dedupe.fit(
        left=train_left,
        right=train_right,
        pairs=y_train,
        verbose=True
    )
    singlegrouper = grouping.SingleGrouping(dedupe=dedupe)
    singlegrouper.findduplicates(data=train_left, n_batches=None, n_records=30)
    singlegrouper.data.to_csv('results_left.csv')
    pass