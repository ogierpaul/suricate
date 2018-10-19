import pandas as pd


class Connector:
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

    def store_new(self, X):
        left_data = X
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

    def upload_data(self, data, gid):
        def update_record(data, ix):
            pass

        def update_group(data, gid):
            pass

        def update_edge(ix, gid):
            pass

    def load_deduplicated(self, new_data, res, aggname):
        """
        # TODO: a retravailler
        script that updates the information in the reference data
        Args:
            new_data: new_data to be loaded in the reference data
            res (list): list of existing results
            aggname (str): ["frequent", "first"]: what group id to take
        """
        existing_gid = calculate_existing_gid(res, aggname=aggname)
        self.upload_data(new_data, gid=existing_gid)
        pass

    def load_results(self, y_results):
        good_matches = get_good_matches(y_results)  # TODO: good matches n'a pas en index les entree orphelines
        return good_matches

    def show_sidebyside(self, left_ix, right_ix, attributes, score=None):
        # TODO
        pass


def get_good_matches(X, left_ix="left_ix", right_ix="right_ix", match="match"):
    """
    Args:
        X (pd.DataFrame) [*(new_ix, ref_ix, pred)]
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
        X[match].astype(bool)
    ].groupBy(
        [left_ix]
    )[right_ix].apply(
        lambda r: r.values.tolist()
    ).reset_index(
        drop=False
    ).toarray()
    return y


def calculate_existing_gid(res, aggname='frequent'):
    """
    evaluation method = most_common
    # TODO: a retravailler avec l'enumeration des gid
    Args:
        res (list): [(ref_ix, *[gid])..]
        aggname (str): ["frequent", "first"]
    Returns:
        str
    """

    assert (aggname in ["frequent", "first"])
    if res is None:
        return None
    elif aggname == "frequent":
        # TODO: use better method of count
        res = dict(
            zip(
                [r[0] for r in res],
                [r[1] for r in res]
            )
        )
        return pd.Series(res).value_counts().iloc[0]
    elif aggname == "first":
        return res[0][1]
    else:
        raise ValueError("value of aggname not recognized")
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
