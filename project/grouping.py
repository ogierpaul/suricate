import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import uuid

conn = psycopg2.connect("host=127.0.0.1 dbname=suricate user=suricateeditor password=66*$%HWqx*")
engine = create_engine('postgresql://suricateeditor:66*$%HWqx*@localhost:5432/suricate')
conn.autocommit = True

df_left = pd.read_sql('SELECT ix, name, street, city, postalcode, countrycode FROM df_source LIMIT 100', con=conn)
df_right = pd.read_sql('SELECT ix, name, street, city, postalcode, countrycode  FROM df_target LIMIT 100', con=conn)
y_pred = pd.read_sql('SELECT ix_left, ix_right, y_pred from results;', con=conn)


def generateuid():
    """

    Returns:
        str: uid hex with 8 digits
    """
    return uuid.uuid4().hex[:8]


def create_distance_matrix(y):
    """
    From the similarity vector *y*, create a symetrical distance Matrix M
    Args:
        y (pd.Series):  similarity vector

    Returns:
        pd.DataFrame
    """
    # Transform the similarity vector to a DataFrame for easier manipulation
    X = pd.DataFrame(y).reset_index()
    ixnamepairs = y.index.names
    # https://stackoverflow.com/questions/22127569/opposite-of-melt-in-python-pandas
    # Pivot the table
    M = X.pivot_table(index=ixnamepairs[0], columns=ixnamepairs[1], values=y.name)

    # Clean the matrix
    ## Fill the missing cols
    missingcols = M.index.difference(M.columns)
    for i in missingcols:
        M[i] = None
        for c in M.loc[i].dropna().index:
            M.loc[c, i] = M.loc[i, c]
    # Fill the blanks
    M = M.fillna(0)

    # Order the columns
    M = M.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    # M should be a distance matrix: so far we have a similarity matrix
    # In M, a score of 0 means two points share the same position, a score of 1 means they are as far apart as it can be
    M = 1 - M
    # M should be symetrical
    M = (M + M.transpose()) / 2
    # And the values in the diagonals should be 0 (identical)
    for i in M.index:
        M.loc[i, i] = 0
    return M


def _utils_show_nearest_neighbours(id_record, y_proba, df_source, ths=0):
    X = pd.DataFrame(y_proba).reset_index()
    X = X.loc[(X['ix_source'] == id_record) & (X['y_proba'] > ths)].sort_values(by='y_proba', ascending=False)
    ixc = X['ix_target'].values
    yp = X['y_proba'].values
    df2 = df_source.loc[ixc].copy()
    df2['y_proba'] = yp
    return df2


def under_threshold(M, threshold=0.25):
    """
    Options to ground this threshold check into solid statistical foundations are welcome.
    Args:
        M (pd.DataFrame): Similarity Matrix
        threshold (float): threshold

    Returns:
        bool: True if the matrix is of length 0 or 1, or if the max of the average distance is below the threshold
    """
    if len(M.index) < 2:
        return True
    else:
        avg_distance = M.mean().max()
        return bool(avg_distance < threshold)


def update_group(M, n_clusters_max=12, threshold=0.5):
    """
    Split the Matrix as much clusters as needed to fall under the threshold
    Args:
        M (pd.DataFrame): Distance Matrix
        n_clusters_max (int): Max number of split possible in the distance matrix
        threshold (float): distance threshold for each cluster, used in under_threshold

    Returns:
        pd.Series: Dictionnary of index to cluster id (uuid)
    """
    if under_threshold(M, threshold):
        y_cluster = pd.Series(index=M.index, data=generateuid())
    else:
        valid_clustering = False
        k = 2
        while valid_clustering == False and k <= n_clusters_max:
            ag = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
            y_cluster = pd.Series(data=ag.fit_predict(M), index=M.index)
            labels = ag.labels_
            subgroups = []
            for c in labels:
                nix = y_cluster.loc[y_cluster == c].index
                r = under_threshold(M.loc[nix, nix], threshold)
                subgroups.append(r)
            valid_clustering = all(subgroups)
            k += 1
        for i in labels:
            y_cluster.loc[y_cluster == i] = generateuid()
    return y_cluster


def cluster_from_matrix(M, y_cluster=None, threshold=0.75):
    """
    Updates the
    Args:
        M (pd.DataFrame): Distance Matrix
        y_cluster (pd.Series): dictionary of cluster ids (uuids). If None, will be initiated.
        threshold (float): distance threshold for coherency of cluster
    Returns
        pd.Series: y_cluster
    """
    if y_cluster is None:
        y_cluster = pd.Series(index=M.index)
    # Init cluster dict
    newpoints = y_cluster.loc[pd.isnull(y_cluster)]
    while newpoints.shape[0] > 0:
        i = newpoints.sample(1).index[0]
        y = M[i].dropna().sort_values(ascending=True)
        y = y.loc[y < threshold]
        goodmatches = y.index
        if i not in goodmatches:
            goodmatches = goodmatches.append(pd.Index([i]))
        existing_gids = y_cluster.loc[goodmatches].dropna()
        if len(existing_gids) == 0:
            newgroup = goodmatches
        else:
            existingix = y_cluster.loc[y_cluster.isin(existing_gids.values)].index
            newgroup = goodmatches.union(existingix).drop_duplicates()
        newclusters = update_group(M.loc[newgroup, newgroup])
        y_cluster.loc[newgroup] = newclusters
        newpoints = y_cluster.loc[pd.isnull(y_cluster)]
    return y_cluster
