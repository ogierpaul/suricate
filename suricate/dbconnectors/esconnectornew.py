from copy import deepcopy
import elasticsearch
import pandas as pd
from suricate.base import ConnectorMixin


ixname = 'ix'
ixname_left = 'ix_left'
ixname_right = 'ix_right'
ixname_pairs = [ixname_left, ixname_right]


class EsConnectorNew:
    def __init__(self, client_es, index_es, doc_type_es, scoreplan, size=30, explain=True,
                 ixname='ix', lsuffix='left', rsuffix='right',
                 es_id='es_id', es_score='es_score', suffix_score='es', es_rank='es_rank'):
        """

        Args:
            client_es (elasticsearch.client): elastic search client
            index_es (str): name of the ES index to search (from GET _cat/indices)
            doc_type_es (str): the name of the document type in the ES database
            scoreplan (dict): list of matches to have
            size (int): max number of hits from ES
            explain (bool): get detailed scores
            ixname (str): default 'ix', index name (in the sense of unique identified of record)
            lsuffix (str): 'left'
            rsuffix (str): 'rigth;
            es_score (str): 'es_score'
            es_id (str): 'es_id'
            suffix_score (str): 'es'
            es_rank (str):'es_rank'
        """
        self.client = client_es
        assert isinstance(client_es, elasticsearch.client.Elasticsearch)
        self.index = index_es
        self.doc_type = doc_type_es
        self.scoreplan = scoreplan
        self.size = size
        self.explain = explain
        self.es_score = es_score
        self.es_rank = es_rank
        self.es_id = es_id
        self.suffix_score = suffix_score
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        # self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
        #     ixname=ixname,
        #     lsuffix=lsuffix,
        #     rsuffix=rsuffix
        # )
        self.usecols = list(self.scoreplan.keys())
        self.outcols = [self.es_score, self.es_rank]
        if self.explain is True:
            self.outcols += [c + '_' + self.suffix_score for c in self.usecols]

    def fetch(self, ix):
        """
        :TODO: to write
        Args:
            ix (pd.Index): index of the records to be passed on

        Returns:
            pd.DataFrame formatted records
        """
        pass

    def fit(self, X=None, y=None):
        """
        Dummy transformer
        Args:
            X:
            y:

        Returns:
            EsConnector
        """
        return self

    def transform(self, X):
        """
        # TODO: to write
        Args:
            X (pd.DataFrame): left data

        Returns:
            np.ndarray: X_score (['ix_left', 'ix_right', 'es_score'])
        """
        alldata = pd.DataFrame(columns=['ix_left', 'ix_right', 'es_score'])
        for lix in X.index:
            record = X.loc[lix]
            res = self.search_record(record)
            score = unpack_allhits(res)
            df = pd.DataFrame(score)
            usecols = X.columns.intersection(df.columns).union(pd.Index([X.index.name]))
            scorecols = pd.Index(['es_rank', 'es_score'])
            df['ix_left'] = lix
            #TODO: We only take score information at the moment
            df.rename(
                columns={
                    'ix': 'ix_right'
                },
                inplace=True
            )
            df = df[['ix_left', 'ix_right', 'es_score']]
            alldata = pd.concat([alldata, df], axis=0, ignore_index=True)
        return alldata.values


    def fit_transform(self, X, y=None):
        self.fit(X=X, y=y)
        return self.transform(X=X)

    def _write_es_query(self, record):
        subqueries = list()
        r2 = record.dropna()
        for inputfield in self.scoreplan.keys():
            if inputfield in r2.index:
                scoretype = self.scoreplan[inputfield]['type']
                subquery = dict()
                if True:
                    subquery = {
                        "match": {
                            inputfield: {
                                "query": record[inputfield],
                                "fuzziness": 2,
                                "prefix_length": 2
                            }
                        }
                    }
                # elif False:
                #     subquery = {
                #         "match": {
                #             inputfield: {
                #                 "query": record[inputfield]
                #             }
                #         }
                #     }
                subqueries.append(subquery)
        mquery = {
            "size": self.size,
            "explain": self.explain,
            "query":
                {
                    "bool": {
                        "should": subqueries
                    }
                }
        }
        return mquery

    def search_record(self, record):
        """

        Args:
            record (pd.Series/dict):

        Returns:
            dict: raw output of ES
        """
        mquery = self._write_es_query(record)
        res = self.client.search(
            index=self.index,
            body=mquery
        )
        return res



def unpack_allhits(res, explain=False, es_id='es_id', es_score='es_score', suffix_score='es', es_rank='es_rank'):
    """
    Args:
        res (dict): raw output of the search engine
        explain (bool): if the results have the _explain field from the explain func of ES
        es_id (str): 'es_id' name of the key for the ES id
        es_score (str): 'es_score' name of the key for the total es score
        suffix_score (str): score for name --> 'name_es'
        es_rank (str): 'es_rank' rank of the match according to ES

    Returns:
        list: list of dicts
    """

    def unpack_onehit(hit, explain=False, es_id='es_id', es_score='es_score', suffix_score='es'):
        """
        For one hit in res['hits']['hits']:
        Get, in a flat dict:
            - the _source field (target data)
            - the total es score (float)

        Args:
            hit (dict):
            explain (bool): if the results have the _explain field from the explain func of ES
            es_id (str): 'es_id' name of the key for the ES id
            es_score (str): 'es_score' name of the key for the total es score
            suffix_score (str): score for name --> 'name_es'

        Returns:
            dict
        """

        def detailedscore(field, n_levels_max=10):
            """
            Drill down from the scores {"value", "description", "details"} \
            to the detailed score
            For a multi-matching query
            Args:
                field (dict): part of the json result file from ES.
                n_levels_max (int):

            Returns:
                dict
            """
            newfield = deepcopy(field)

            value = None
            on_col = None
            if type(field) == dict and field.get('value') is not None:
                value = field.get(value)
            for i in range(n_levels_max):
                if type(newfield) == dict:
                    if newfield.get('description') is not None and 'weight' in newfield['description']:
                        if value is None:
                            value = newfield['value']
                        on_col = newfield['description'][7:].split(':')[0]
                        break
                    else:
                        if newfield.get('details') is not None:
                            newfield = newfield['details']
                        else:
                            return None
                elif type(newfield) == list:
                    if len(newfield) > 0:
                        newfield = newfield[0]
                    else:
                        return None
            return value, on_col

        sd = dict()
        sd[es_id] = hit['_id']
        sd.update(hit['_source'])
        sd[es_score] = hit['_score']
        if explain is True:
            if hit.get('_explanation') is not None:
                for field in hit['_explanation']['details']:
                    a = detailedscore(field)
                    if a is not None:
                        (value, col) = a
                        if value is not None:
                            scorename = '_'.join([col, suffix_score])
                            sd[scorename] = value
        return sd

    allhits = res['hits']['hits']
    results = []
    for i, hit in enumerate(allhits):
        sd = unpack_onehit(hit, explain=explain, es_id=es_id, es_score=es_score, suffix_score=suffix_score)
        sd[es_rank] = i
        results.append(sd)
    return results

