import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing.imputation import Imputer

from operations import companypreparation as preprocessing
from wookie import comparators, connectors

if __name__ == '__main__':
    # Variable definition
    ## indexes
    ixname = 'ix'
    lsuffix = '_left'
    rsuffix = '_right'
    ixleft = ixname + lsuffix
    ixright = ixname + rsuffix
    ixs = [ixleft, ixright]
    ## File path
    filepath_left = 'T2 Signalis Frozen Supplier list for P11 Vendor search_17Sept2018.xlsx'
    filepath_right = 'liste fr test pour trouver doublons dans SAP.xlsx'
    filepath_training = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'
    filepath_results = 'results from {}.xlsx'.format(pd.datetime.now().strftime("%d-%m-%y %Hh%M"))
    ## Estimator
    n_estimators = 2000  # number of estimators for the Gradient Boosting Classifier
    displaycols = [
        'name',
        'street',
        'postalcode',
        'city',
        'countrycode',
        'siret',
        'siren',
        'euvat',
        'kapis',
        'duns'
    ]

    # Data Preparation
    ## Prepare df_left
    newleft = pd.read_excel(
        filepath_left
    ).rename(
        columns={
            'supplierid': ixname,
            'country_code': 'countrycode',
            'eu_vat': 'euvat'
        }
    ).drop(
        [
            'Existing\nP11\nSupplier',
            'P11 Supplier\nName'
        ],
        axis=1
    )
    namecols = ['name1', 'name2', 'name3', 'name4']
    newleft['name'] = newleft.apply(
        lambda r: preprocessing.concatenate_names(r[namecols]),
        axis=1
    )
    newleft.drop(namecols, axis=1, inplace=True)
    del namecols

    streetcols = ['street1', 'street2']
    newleft['street'] = newleft.apply(
        lambda r: preprocessing.concatenate_names(r[streetcols]),
        axis=1
    )
    newleft.drop(streetcols, axis=1, inplace=True)
    newleft.set_index([ixname], inplace=True)
    newleft = preprocessing.preparedf(newleft, ixname=ixname)

    ## Prepare df_right
    newright = pd.read_excel(
        filepath_right
    ).rename(
        columns={
            'supplierid': ixname,
            'country_code': 'countrycode',
            'eu_vat': 'euvat'
        }
    ).drop(
        [
            'Code\nPost',
            'Dept',
            'Pays',
            'Length\nSAP',
            'Postal\nCode\nrequired',
            'Nom\nPays',
            'Nbr caract\nN° VAT\nOPALE',
            'Length\nSAP P11',
            'VAT nb mandatory if Europe ctry',
            'NAF',
            'Telephone'

        ],
        axis=1
    )
    namecols = ['name1', 'name2']
    newright['name'] = newright.apply(
        lambda r: preprocessing.concatenate_names(r[namecols]),
        axis=1
    )
    newright.drop(namecols, axis=1, inplace=True)
    del namecols

    streetcols = ['street1', 'street2']
    newright['street'] = newright.apply(
        lambda r: preprocessing.concatenate_names(r[streetcols]),
        axis=1
    )
    newright.set_index([ixname], inplace=True)
    newright = preprocessing.preparedf(newright, ixname=ixname)
    newright['siren'] = newright['siret'].apply(lambda r: None if pd.isnull(r) else r[:9])
    newright.drop(streetcols, axis=1, inplace=True)

    ## Load x_train and y_train

    df_train = pd.read_csv(
        filepath_training
    ).rename(
        columns={
            'zip_left': 'postalcode_left',
            'zip_right': 'postalcode_right',
            'country_left': 'countrycode_left',
            'country_right': 'countrycode_right'
        }
    )
    for c in ixs:
        df_train[c] = df_train[c].apply(preprocessing.idtostr)
    df_train.set_index(ixs, inplace=True)

    # concat and train the data
    X_left, X_right, pairs = connectors.concattrainnew(left=newleft, right=newright, trainsbs=df_train,
                                                       func=preprocessing.preparedf)

    # CONNECTOR SCORES
    idcols = ['duns', 'siret', 'siren', 'kapis', 'euvat']
    namescore = comparators.tfidfconnector(left=X_left, right=X_right, on='name_ascii', threshold=0,
                                           stop_words=preprocessing.companystopwords)
    streetscore = comparators.tfidfconnector(left=X_left, right=X_right, on='street_ascii', threshold=0,
                                             stop_words=preprocessing.streetstopwords)
    idsscore = comparators.idsconnector(left=X_left, right=X_right, on_cols=idcols)
    connectscores = comparators.mergescore([namescore, streetscore, idsscore])

    # SIDE BY SIDE SCORES
    X_sbs = connectors.createsbs(pairs=connectscores.reset_index(drop=False), left=X_left, right=X_right)
    ixall = X_sbs.index
    ixtrain = pairs.index
    X_train = X_sbs.loc[ixtrain]
    y_train = pairs['y_true']
    X_new = X_sbs.loc[
        ~(
            (
                ixall.get_level_values(ixleft).isin(ixtrain.get_level_values(ixleft))
            ) | (
                ixall.get_level_values(ixright).isin(ixtrain.get_level_values(ixright))
            )
        )
    ]
    ixnew = X_new.index

    # PIPELINE
    ## Scores
    sbsscores = comparators.PipeComparator(
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
    )
    dp_connectscores = comparators.DataPasser(on_cols=connectscores.columns.tolist())
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
    print('model score on training data:{}'.format(trainscore))

    # Calculate and format the results
    y_pred = model.predict(X_new)
    y_pred = pd.Series(y_pred, index=ixnew)
    goodmatches = pd.DataFrame(y_pred.loc[y_pred == 1])
    results = connectors.showpairs(pairs=goodmatches, left=newleft, right=newright, usecols=displaycols)
    results.to_excel(filepath_results, index=True)
    pass
