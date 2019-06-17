from copy import deepcopy

import elasticsearch
import pandas as pd


from suricate.preutils.indextools import addsuffix, concatixnames

ixname = 'ix'
ixname_left = 'ix_left'
ixname_right = 'ix_right'
ixname_pairs = [ixname_left, ixname_right]


class EsConnector:
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
        self.ixnameleft, self.ixnameright, self.ixnamepairs = concatixnames(
            ixname=ixname,
            lsuffix=lsuffix,
            rsuffix=rsuffix
        )
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
            pd.DataFrame: X_score
        """

        for lix in X.index:
            record = X.loc[lix]
            res = self.search_record(record)
            df = pd.DataFrame(res)
            df['ix_left'] = lix

        pass

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

    def search_serie(self, left):
        """
        Args:
            left (pd.DataFrame/pd.Series):
        Returns:
            pd.DataFrame: {['ix_left', 'ix_right', relevance_scores, name, ...]}
        """
        tempcol1 = 'f1b3'
        tempcol2 = 'f2b4'
        # Format series for use as pd.DataFrame
        if left is pd.Series():
            newleft = pd.DataFrame(left).transpose()
            assert newleft.shape[0] == 1
        else:
            newleft = left
        # - Launch the search
        # - unpack results
        # - format as dataframe
        # - melt to have all results side by side
        df_res = newleft.apply(
            lambda r: self.search_record(record=r),
            axis=1
        ).apply(
            lambda r: unpack_allhits(res=r, explain=self.explain)
        ).apply(
            pd.Series
        ).reset_index(
            drop=False
        ).rename(
            {self.ixname: self.ixnameleft},
            axis=1
        )
        fres = pd.melt(
            df_res,
            id_vars=[self.ixnameleft],
            var_name=tempcol1,
            value_name=tempcol2
        ).drop(
            [tempcol1],
            axis=1
        )
        fres2 = fres[
            tempcol2
        ].apply(
            pd.Series
        ).rename(
            {self.ixname: self.ixnameright},
            axis=1
        )
        fres2[self.ixnameleft] = fres[self.ixnameleft]
        return fres2

    def lrsearch(self, left):
        """
        use a dataframe and return the results formatted as a dataframe
        With left-right, SBS-Style
        Args:
            left (pd.DataFrame):

        Returns:
            pd.DataFrame
        """
        res = self.search_serie(
            left=left
        )
        # Deal with possible missing columns
        for c in self.outcols:
            if not c in res:
                res[c] = None
        # Filter on used cols (score, index, and used for sbs) and rename use_cols with '_right'
        sbs = res.loc[:, self.ixnamepairs + self.usecols + self.outcols]
        renamecols = list(
            filter(
                lambda x: x not in self.ixnamepairs + self.outcols,
                sbs.columns
            )
        )
        sbs.rename(
            columns=dict(
                zip(
                    renamecols,
                    map(
                        lambda r: '_'.join([r, self.rsuffix]),
                        renamecols

                    ),

                )
            ),
            inplace=True
        )
        # rename left with '_left'
        newleft = addsuffix(df=left[self.usecols], suffix=self.lsuffix)
        sbs = pd.merge(
            left=sbs,
            right=newleft,
            left_on=self.ixnameleft,
            right_index=True,
            how='inner'
        ).set_index(
            self.ixnamepairs
        )
        return sbs



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

