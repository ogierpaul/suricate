import pandas as pd
from py2neo import Graph, Node

hid_col = ['hid']
label = 'SupplierID'
name_cols = ['name']
location_cols = ['street', 'postalcode', 'city', 'countrycode']
entity_cols = ['duns']


def write_merge_node(label, merge_on, properties=None):
    """
    This function write the query to push a row as a Node
    Args:
        label (str): name of the label of the node in neo4j
        merge_on (str): name of the attribute on which to merge
        properties (list): name of the attributes to be passed on to the node.
    Returns:
        str: query
    Examples:
        write_query(label='Person',merge_on='name',attributes=['age','eye_color','gender'])

            MERGE (n:Person{name :$name})
            ON CREATE SET
                 n.age = $age,
                 n.eye_color = $eye_color,
                 n.nationality = $nationality;
    """

    if merge_on is not None:
        query = """
        MERGE (n:{label}{{{merge_on}:${merge_on}}})
        ON CREATE SET
        """.format(label=label, merge_on=merge_on)
    else:
        query = """MERGE (n:{label})
        ON CREATE SET
        """.format(label=label, merge_on=merge_on)
    for c in properties:
        query += '\t n.{0} = ${0},\n'.format(c)
    query = query.rstrip('\n')
    query = query.rstrip(',')
    query += ';'
    return query


def write_merge_edge(start_label, start_id, target_label, target_id, reltype, attributes):
    # TODO : DOC
    if start_label is not None:
        query_start = """(a:{start_label}{{{start_id}:${start_id}}})""".format(start_label=start_label,
                                                                               start_id=start_id)
    else:
        query_start = """(a{{{start_id}:${start_id}}})""".format(start_id=start_id)

    if target_label is not None:
        query_target = """(b:{target_label}{{{target_id}:${target_id}}})""".format(target_label=target_label,
                                                                                   target_id=target_id)
    else:
        query_target = """(b{{{target_id}:${target_id}}})""".format(target_label=target_label, target_id=target_id)
    query_rel = """-[r:${reltype}]->""".format(reltype=reltype)
    query = "MERGE " + query_start + query_rel + query_target
    query += "\n ON CREATE SET \n"
    for c in attributes:
        query += '\t r.{0} = ${0},\n'.format(c)
    query = query.rstrip('\n')
    query = query.rstrip(',')
    query += ';'
    return query


class Merger:
    def __init__(self, graph):
        """

        Args:
            graph (py2neo.Graph): target graph
        """
        self.graph = graph
        self._uploadeds = list()
        self.df = None

    def _merge_nodes(self, df, label, merge_on, attributes):
        """
        This function creates a Node in neo4j for each row in this DataFrame.
        The DataFrame will be pushed in one commit.
        Args:
            df (pd.DataFrame): data to be pushed
            label (str): name of the label of the node in neo4j
            merge_on (str): name of the attribute on which to merge
            attributes (list): name of the attributes to be passed on to the node, they are columns in the dataframe

        Returns:
            None

        Examples:
            df=pd.DataFrame([{'name':'Alice','age':20,'eye_color':'Blue','nationality':'Swede'},
                     {'name':'Bob','age':40,'eye_color':'Brown','nationality':'American'}])
            will be uploaded with the query:

            "MERGE (n:Person{name :$name})
            ON CREATE SET
                n.age = $age,
                n.eye_color = $eye_color,
                n.nationality = $nationality;"

        """
        query = write_merge_node(label, merge_on, attributes)
        self.query = query

        tx = self.graph.begin()
        for ix, row in df.iterrows():
            params = {merge_on: row[merge_on]}
            for c in attributes:
                params[c] = row[c]
            tx.evaluate(query, parameters=params)
        tx.commit()
        return None

    def merge_nodes(self, df, label, merge_on, attributes, verbose=False, batch_size=10000):
        """
        This function runs several time the push_dataframe function, per batch_size (number of rows) for each commit
        It allows to have several commites
        Args:
        Args:
            df (pd.DataFrame): data to be pushed
            label (str): name of the label of the node in neo4j
            merge_on (str): name of the attribute on which to merge
            attributes (list): name of the attributes to be passed on to the node, they are columns in the dataframe
            verbose (bool): give an up_to_date status if the upload
            batch_size (int): default 10000, if None, all uploaded in one single transaction
        Returns:
            pd.DataFrame: the dataframe itself, with a new column 'committed' for each rows that have been successfully commited
        """

        uploadfunction = lambda x: self._merge_nodes(df=x, label=label, merge_on=merge_on, attributes=attributes)

        self._mergebatch(df=df, uploadfunction=uploadfunction, verbose=verbose, batch_size=batch_size)

        return None

    def _merge_edges(self, df, start_label, start_id, target_label, target_id, reltype, attributes):
        query = write_merge_edge(start_label, start_id, target_label, target_id, reltype, attributes)
        self.query = query

        tx = self.graph.begin()
        for ix, row in df.iterrows():
            params = dict()
            for c in [start_label, start_id, target_label, target_id, reltype]:
                params[c] = row[c]
            for c in attributes:
                params[c] = row[c]
            tx.evaluate(query, parameters=params)
        tx.commit()
        return None

    def merge_edges(self, df, start_label, start_id, target_label, target_id, reltype, attributes, verbose=False,
                    batch_size=10000):
        """
        This function runs several time the push_dataframe function, per batch_size (number of rows) for each commit
        It allows to have several commites
        Args:
        Args:
            df (pd.DataFrame): data to be pushed
            start_label (str)
            start_id
            target_label
            target_id
            reltype (str)
            attributes (list): name of the attributes to be passed on to the node, they are columns in the dataframe
            verbose (bool): give an up_to_date status if the upload
            batch_size (int): default 10000, if None, all uploaded in one single transaction
        Returns:
            pd.DataFrame: the dataframe itself, with a new column 'committed' for each rows that have been successfully commited
        """

        uploadfunction = lambda x: self._merge_edges(
            df=x, start_label=start_label, start_id=start_id,
            target_label=target_label, target_id=target_id,
            reltype=reltype, attributes=attributes)

        self._mergebatch(df=df, uploadfunction=uploadfunction, verbose=verbose, batch_size=batch_size)

        return None

    def _mergebatch(self, df, uploadfunction, verbose, batch_size):
        if batch_size is None:
            batch_size = df.shape[0]

        self.df = df
        todo = self._select_next(batch_size=batch_size)
        i = 0

        while todo.shape[0] > 0:
            i += 1
            if verbose:
                print(
                    'batch {}:{} rows left at {}'.format(i, self.df.loc[~self.df.index.isin(self._uploadeds)].shape[0],
                                                         pd.datetime.now()))
            uploadfunction(todo)

            # if push is successful
            self._update_commited(new=list(todo.index))
            todo = self._select_next(batch_size=batch_size)

        if todo.shape[0] == 0:
            if verbose:
                print('merged successfuly all {} rows'.format(len(self._uploadeds)))
            self._uploadeds = list()

        return None

    def _update_commited(self, new):
        """

        Args:
            new (list): list of indexes successfully committed

        Returns:
            None
        """
        self._uploadeds += list(new)
        return None

    def _select_next(self, batch_size):
        possibles = self.df.loc[~self.df.index.isin(self._uploadeds)]
        n = min(batch_size, possibles.shape[0])
        possibles = possibles.iloc[:n]
        return possibles


class NeoConnector:
    def __init__(self,
                 hid_col,
                 group_col,
                 graph,
                 name_cols=None,
                 entity_cols=None,
                 location_cols=None,
                 label='SupplierID'
                 ):
        """

        Args:
            hid_col (str):
            group_col (str):
            graph (Graph):
            name_cols (list):
            entity_cols (list):
            location_cols (list):
        """
        for c in hid_col, group_col, label:
            assert isinstance(c, str)
        self.hid_col = hid_col
        self.group_col = group_col
        self.label = label
        for c in [name_cols, entity_cols, location_cols]:
            if not c is None:
                assert hasattr(c, '__iter__')
        self.name_cols = name_cols
        self.entity_cols = entity_cols
        self.location_cols = location_cols
        assert isinstance(g, Graph)
        self.graph = graph
        pass

    def upload(self, record):
        node = Node()
        pass
