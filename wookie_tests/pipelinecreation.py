import pandas as pd
from sklearn.pipeline import make_pipeline

# TODO: what if left_index was omitted in the connector phase? (No relevant proposition found in right reference data)
# The connector should display None to the DataFrame? --> Consequences?
left_data = {
    "index": [0, 1, 2]
    "name": ['foo', 'bar', 'ninja']
}
right_data = {
    "index": [0, 1, 2]
    "name": ['foo', 'bar', 'baz'],
    "gid": ['0a', '1b', '1b']
}

# The side by side should be with None
sidebyside = [
    [(0, "foo"), (0, "foo")]  # case equal
    [(0, "foo"), (1, "bar")]  # case False
    [(1, "bar"), (1, "bar")]  # case True for two
    [(1, "bar"), (1, "baz")]  # case True for two fuzzy
    [(2, "ninja"), None]  # case None --> To be removed
]
expected_predictions = [1, 0, 1, 1]  # only first four cases of the sidebyside

good_matches = [
    (0, [0]),
    (1, [1, 2]),
    (2, None)
]
gids = [
    (0, '0a')
    (1, '1b'),
    (2, None)
]


class Connector()
    def fetch_reference(self, ix, attributes=None):
        """
        Args:
            ix (scalar): index (to do: add as list)
            attributes (list): list of attributes
        Returns:
            dict json-like format of attributes
        """
        res = ix
        return ix  # Why not use yield?

    def store_new(self, X)
        self.left_data = X
        pass

    def fetch_new(self, ix, attributes=None):
        """
        Args:
            ix (scalar):
            attributes (list):
        Returns:
            dict json-like format of attributes
        """
        return ix

    def upload_data(data, gid):
        def update_record(data, ix):
            pass

        def update_group(data, gid):
            pass

        def update_edge(ix, gid):
            pass

    def load_deduplicated(new_data, res, existing):
        """
        # TODO: a retravailler
        script that updates the information in the reference data
        Args:
            new_data: new_data to be loaded in the reference data
            res (list): list of existing results
            existing (str): ["frequent", "first"]: what group id to take
        """
        existing_gid = calculate_existing_gid(res, use=existing)
        upload_data(new_data, gid=existing_gid)
        pass

    def load_results(y_results):
        good_matches = get_good_matches(y_results)  # TODO: good matches n'a pas en index les entree orphelines
        return res

    def show_sidebyside(left_ix, right_ix, attributes, score=None):
        # TODO
        pass


con = Connector()  # connector should return Side by Side
model = Estimator()  # estimator should return expected result
index_keeper = []
model.fit(X_train, y_train)
dp = DataPasser()

X_results = make_pipeline(
    *[con, est]
)
pipe.fit(X_train, y_train)
X_results = pipe.transform(X_new)
y_results = list_goodmatches(X_results)
y_gid = map(
    lambda r: calculate_existing_gid(r), y_results
)
new_data = Connector.load_deduplicated(X_new, y_gid)


# output of X_results:
# [*[*(ref_ix, gid)]]

def calculate_existing_gid(res, use='frequent'):
    """
    evaluation method = most_common
    # TODO: a retravailler avec l'enumeration des gid
    Args:
        res (list): [(ref_ix, *[gid])..]
        use (str): ["frequent", "first"]
    Returns:
        str
    """

    assert (use in ["frequent", "first"])
    if res is None:
        return None
    elif use == "frequent":
        # TODO: use better method of count
        res = dict(
            zip(
                [r[0] for r in res],
                [r[1] for r in res]
            )
        )
        return pd.Series(res).value_counts().iloc[0]
    elif use == "first":
        return res[0][1]
    else:
        RaiseError("value of use not recognized")
        pass


def create_gid(m):
    """
    create a group id from several parameters
    """
    # TODO: create a hashing method to create a gid
    gid = m
    return gid


def create_group_record(d):
    """
    create a group record data from the new data
    Args:
        d (dict): new data
    Returns:
        dict : new group record
    """
    r = d
    return d


def get_good_matches(self, X):
    """
    Args:
        X (pd.DataFrame)
    Returns:
        y (list)
    Examples:
        X = pd.DataFrame(
            {'left_ix': [0, 0, 1, 1]},
            {'right_ix': [1, 2, 3, 4]},
            {'matches: [1, 0, 1, 1]}
        )
        get_good_matches(X):
            [
                (0, [1]),
                (1, [3, 4])
            ]
    """
    y = X.copy(
    ).loc[
        df[self.match].astype(bool)
    ].groupBy(
        ["left_ix"]
    )['right_ix'].apply(
        lambda r: r.values.tolist()
    ).reset_index(
        drop=False
    ).toarray()
    return y


training_path = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/wcmodel.csv'
X = pd.read_csv(training_path).rename(columns={
    'wc_true': 'y_true'
})
y = X['y_true']
X = X[list(filter(lambda x: x != 'y_true', X.columns))]
stopwords = ['gmbh', 'ltd', 'hotel', 'co', 'kg', 'ag']

comp1 = PipeComparator(
    scoreplan={
        'name': ['exact', 'fuzzy', 'token'],
        'street': ['exact', 'token'],
        'duns': ['exact'],
        'city': ['fuzzy'],
        'postalcode': ['exact'],
        'country_code': ['exact']
    }
)
tokenizer_name = TfidfVectorizer(stop_words=stopwords)
tfname = TokenComparator(
    train_col='name_right',
    new_col='name_left',
    train_ix='ix_right',
    new_ix='ix_left',
    tokenizer=tokenizer_name
)

tokenizer_street = TfidfVectorizer()
tfstreet = TokenComparator(
    train_col='street_right',
    new_col='street_left',
    train_ix='ix_right',
    new_ix='ix_left',
    tokenizer=tokenizer_street
)
scores = make_union(
    *[
        comp1,
        tfname,
        tfstreet
    ]

)
estimator = RandomForestClassifier(n_estimators=500)
e2epipe = make_pipeline(
    *[
        scores,
        estimator
    ]
)
