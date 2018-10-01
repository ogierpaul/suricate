import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing.imputation import Imputer

from operations import companypreparation as preprocessing
from wookie import comparators, connectors

ixname = 'ix'
lsuffix = '_left'
rsuffix = '_right'
ixs = [ixname + lsuffix, ixname + rsuffix]

# Prepare df_left
newleft = pd.read_excel(
    'T2 Signalis Frozen Supplier list for P11 Vendor search_17Sept2018.xlsx'
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

# Prepare df_right
newright = pd.read_excel('liste fr test pour trouver doublons dans SAP.xlsx').rename(
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

# Load x_train and y_train
trainingpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'
df_train = pd.read_csv(trainingpath).rename(
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
X_train = X_sbs.loc[pairs.index]
y_train = pairs['y_true']

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
        'postalcode_1char': ['exact'],
        'postalcode_2char': ['exact'],
        'countrycode_ascii': ['exact']
    }
)
dp_connectscores = comparators.DataPasser(on_cols=connectscores.columns.tolist())
scorer = make_union(
    *[
        sbsscores,
        dp_connectscores
    ]
)

## Estimator
estimator = GradientBoostingClassifier(n_estimators=10)
imp = Imputer()
model = make_pipeline(
    *[
        scorer,
        imp,
        estimator
    ]
)

# model.fit(X_train, y_train)
# trainscore = model.score(X_train, y_train)
# print('model score on training data:{}'.format(trainscore))
