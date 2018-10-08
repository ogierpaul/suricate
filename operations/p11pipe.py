import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing.imputation import Imputer

import wookie.oaacomparators
from operations import companypreparation as preprocessing
from wookie import sbscomparators, connectors

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
    n_estimators = 10  # number of estimators for the Gradient Boosting Classifier
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
    left = pd.read_csv(filepath_left, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    right = pd.read_csv(filepath_right, sep=',', encoding='utf-8', dtype=str, nrows=50).set_index(ixname)
    df_train = pd.read_csv(filepath_training).set_index(ixnamepairs)
    train_left, train_right, y_train = connectors.separatesides(df_train)



    # concat and train the data
    X_left, X_right, pairs = connectors.concattrainnew(
        left=left,
        right=right,
        trainsbs=df_train,
        transfunc=lambda df: preprocessing.preparedf(data=df, ixname=ixname)
    )

    # CONNECTOR SCORES
    idcols = [
        'duns'
        # , 'siret', 'siren', 'kapis', 'euvat'
    ]

    namescore = wookie.oaacomparators.tfidfconnector(left=X_left, right=X_right, on='name_ascii', threshold=0,
                                                     stop_words=preprocessing.companystopwords)
    streetscore = wookie.oaacomparators.tfidfconnector(left=X_left, right=X_right, on='street_ascii', threshold=0,
                                                       stop_words=preprocessing.streetstopwords)
    idsscore = wookie.oaacomparators.idsconnector(left=X_left, right=X_right, on_cols=idcols)
    connectscores = wookie.oaacomparators.mergescore([namescore, streetscore, idsscore])

    # SIDE BY SIDE SCORES
    X_sbs = connectors.createsbs(pairs=connectscores, left=X_left, right=X_right)
    ixall = X_sbs.index
    ixtrain = pairs.index
    # TODO: What to do of missing data
    ixtrain = ixtrain.intersection(X_sbs.index)
    X_train = X_sbs.loc[ixtrain]
    y_train = pairs.loc[ixtrain, 'y_true']
    X_new = X_sbs.loc[
        ~(
            (
                ixall.get_level_values(ixnameleft).isin(ixtrain.get_level_values(ixnameleft))
            ) | (
                ixall.get_level_values(ixnameright).isin(ixtrain.get_level_values(ixnameright))
            )
        )
    ]
    ixnew = X_new.index

    # PIPELINE
    ## Scores
    sbsscores = sbscomparators.PipeSbsComparator(
        scoreplan={
            # 'name_ascii': ['exact', 'fuzzy', 'token'],
            # 'street_ascii': ['exact', 'token'],
            # 'street_ascii_wostopwords': ['token'],
            # 'name_ascii_wostopwords': ['fuzzy'],
            # 'city': ['fuzzy'],
            # 'postalcode_ascii': ['exact'],
            # 'postalcode_2char': ['exact'],
            'countrycode': ['exact']
        }
    )
    dp_connectscores = sbscomparators.DataPasser(on_cols=connectscores.columns.tolist())
    # noinspection PyTypeChecker
    scorer = make_union(
        *[
            sbsscores,
            dp_connectscores
        ]
    )

    ## Estimator
    estimator = GradientBoostingClassifier(n_estimators=n_estimators)
    imp = Imputer()
    model = make_pipeline(
        *[
            scorer,
            imp,
            estimator
        ]
    )

    model.fit(X_train, y_train)
    trainscore = model.score(X_train, y_train)
    print('model score on training data: {}'.format(trainscore))


    # Calculate and format the results
    y_pred = model.predict(X_new)
    y_pred = pd.Series(y_pred, index=ixnew)
    goodmatches = pd.DataFrame(y_pred.loc[y_pred == 1])
    results = connectors.showpairs(pairs=goodmatches, left=left, right=right, usecols=displaycols)
    # results.to_excel(filepath_results, index=True)
    pass
