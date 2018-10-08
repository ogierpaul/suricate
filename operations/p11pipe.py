import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from operations import companypreparation as preprocessing
from wookie import sbscomparators, connectors, interface, oaacomparators

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
    n_estimators = 2000  # number of estimators for the Gradient Boosting Classifier
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

    sbs = interface.SbsModel(
        prefunc=lambda df: preprocessing.preparedf(df, ixname='ix'),
        idtfidf=oaacomparators.IdTfIdfConnector(
            id_cols=['euvat', 'siret'],
            tfidf_cols=['name', 'street'],
            stop_words={
                'name': preprocessing.companystopwords,
                'street': preprocessing.streetstopwords
            }
        ),
        sbscomparator=sbscomparators.PipeSbsComparator(
            scoreplan={
                'name_ascii': ['exact', 'fuzzy', 'token'],
                'street_ascii': ['exact', 'token'],
                'street_ascii_wostopwords': ['token'],
                'name_ascii_wostopwords': ['fuzzy'],
                'city': ['fuzzy'],
                'postalcode_ascii': ['exact'],
                'postalcode_2char': ['exact'],
                'countrycode': ['exact']
            }
        ),
        estimator=GradientBoostingClassifier(n_estimators=n_estimators)
    )
    print(pd.datetime.now(), 'start')
    sbs.fit(
        left=train_left,
        right=train_right,
        pairs=y_train
    )
    print(pd.datetime.now(), 'stop fit')
    score = sbs.score(train_left, train_right, y_train)
    print(pd.datetime.now(), 'score: ', score)
    pass
