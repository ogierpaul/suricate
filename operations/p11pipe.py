import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from operations import companypreparation as preprocessing
from wookie import connectors, LrDuplicateFinder

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
    filepath_training = 'data/trainingdata.csv'
    filepath_results = 'results from {}.xlsx'.format(pd.datetime.now().strftime("%d-%m-%y %Hh%M"))
    ## Estimator
    n_estimators = 5  # number of estimators for the Gradient Boosting Classifier
    displaycols = [
        'name',
        'street',
        'postalcode',
        'city',
        'countrycode',
        # 'siret',
        # 'siren',
        # 'euvat'
    ]

    # Data Preparation
    # left = pd.read_csv(filepath_left, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    # right = pd.read_csv(filepath_right, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    df_train = pd.read_csv(filepath_training).set_index(ixnamepairs)
    train_left, train_right, y_train = connectors.separatesides(df_train)

    sbs = LrDuplicateFinder(
        prefunc=preprocessing.preparedf,
        scoreplan={
            'name': {
                'type': 'FreeText',
                'stop_words': preprocessing.companystopwords,
                'threshold': 0.6,
            },
            'street': {
                'type': 'FreeText',
                'stop_words': preprocessing.streetstopwords,
                'threshold': 0.6
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
        estimator=GradientBoostingClassifier(n_estimators=n_estimators),
        verbose=True
    )
    print(pd.datetime.now(), 'start')
    sbs.fit(
        left=train_left,
        right=train_right,
        pairs=y_train,
    )
    print(pd.datetime.now(), 'stop fit')
    score = sbs.score(train_left, train_right, y_train)
    print(pd.datetime.now(), 'score: ', score)
    pass
