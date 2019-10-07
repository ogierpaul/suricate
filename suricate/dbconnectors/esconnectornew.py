from copy import deepcopy
import elasticsearch
import pandas as pd
from suricate.base import ConnectorMixin
from suricate.lrdftransformers.cartesian import create_lrdf_sbs
import numpy as np

ixname = 'ix'
ixname_left = 'ix_left'
ixname_right = 'ix_right'
ixname_pairs = [ixname_left, ixname_right]


class EsConnectorNew(ConnectorMixin):
    """
    Elastic Search Connector for the suricate project
    Attributes:
        client (elasticsearch.client.Elasticsearch):
        ixname:
        lsuffix:
    """
    def __init__(self, client, index, scoreplan,   doc_type='_doc', size=30, explain=True,
                 ixname='ix', lsuffix='left', rsuffix='right',
                 es_id='es_id', es_score='es_score', suffix_score='es', es_rank='es_rank'):
        """

        Args:
            client (elasticsearch.client.Elasticsearch): elastic search client
            index (str): name of the ES index to search (from GET _cat/indices)
            doc_type (str): the name of the document type in the ES database
            scoreplan (dict): list of matches to have (see above)
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
        ConnectorMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix)
        self.client = client
        assert isinstance(self.client, elasticsearch.client.Elasticsearch)
        self.index = index
        self.doc_type = doc_type
        self.scoreplan = scoreplan
        self.size = size
        self.explain = explain
        self.es_score = es_score
        self.es_rank = es_rank
        self.es_id = es_id
        self.suffix_score = suffix_score
        self.usecols = list(self.scoreplan.keys())
        self.outcols = [self.es_score, self.es_rank]
        if self.explain is True:
            self.outcols += [c + '_' + self.suffix_score for c in self.usecols]

    def fetch_left(self, X, ix):
        """
        Args:
            X: input data (left)
            ix (pd.Index): index of the records to be passed on

        Returns:
            pd.DataFrame formatted records
        """
        return X.loc[ix]

    def fetch_right(self, ix=None, X=None):
        """
        :TODO: to test
        Args:
            X: dummy, input data to be given to the connector
            ix (pd.Index): index of the records to be passed on NEEDED

        Returns:
            pd.DataFrame formatted records
        """
        results={}
        for i in ix.values:
            assert isinstance(self.client, elasticsearch.client.Elasticsearch)
            res = self.client.get(index=self.index, id=i, doc_type=self.doc_type)
            if res['found'] is False:
                raise IndexError(
                    'id: {} not found in ES Index {} for doc_type {}'.format(i, self.index, self.doc_type)
                )
            else:
                data = res['_source']
                results[i] = data
        X = pd.DataFrame.from_dict(data=results, orient='index')
        # If we have a duplicate column ix
        if X.index.name in X.columns:
            X.drop(labels=[X.index.name], axis=1, inplace=True)
        return X



    def getindex(self, X):
        """
        TODO: to write
        Args:
            X:

        Returns:
            pd.MultiIndex
        """
        pass

    def getsbs(self, X, on_ix=None):
        """
        TODO: to test
        Args:
            X (pd.DataFrame): input data
            on_ix (pd.MultiIndex):

        Returns:
            pd.DataFrame
        """
        ix_left = np.unique(on_ix.get_level_values(self.ixnameleft))
        ix_right = np.unique(on_ix.get_level_values(self.ixnameright))
        left = self.fetch_left(X=X, ix=ix_left)
        right = self.fetch_right(ix=ix_right)
        df = create_lrdf_sbs(X=[left, right], on_ix=on_ix, ixname=self.ixname, lsuffix=self.lsuffix, rsuffix=self.rsuffix)
        return df

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
        # TODO: check use of suffixes and column names
        # TODO: We only take score information at the moment
        Args:
            X (pd.DataFrame): left data

        Returns:
            pd.DataFrame: X_score ({['ix_left', 'ix_right'}: 'es_score'])
        """
        alldata = pd.DataFrame(columns=[self.ixnameleft, self.ixnameright, self.es_score, self.es_rank])
        for lix in X.index:
            record = X.loc[lix]
            res = self.search_record(record)
            score = unpack_allhits(res)
            df = pd.DataFrame(score)
            scorecols = pd.Index([self.es_score, self.es_rank])
            df[self.ixnameleft] = lix
            df.rename(
                columns={
                    'ix': self.ixnameright
                },
                inplace=True
            )
            df = df[[self.ixnameleft, self.ixnameright, 'es_score']]
            alldata = pd.concat([alldata, df], axis=0, ignore_index=True)
        Xt = alldata.set_index(self.ixnamepairs)
        return Xt


    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y)
        return self.transform(X=X)

    def _write_es_query(self, record):
        subqueries = list()
        r2 = record.dropna()
        for inputfield in self.scoreplan.keys():
            if inputfield in r2.index:
                subquery = {
                    "match": {
                        inputfield: {
                            "query": record[inputfield],
                            "fuzziness": 2,
                            "prefix_length": 2
                        }
                    }
                }
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

