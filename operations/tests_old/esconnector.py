from copy import deepcopy

import elasticsearch
import pandas as pd
import psycopg2 as pg

from wookie.dbconnectors import addsuffix
from wookie.preutils import concatixnames, name_freetext, name_exact

ixname = 'ix'
ixname_left = 'ix_left'
ixname_right = 'ix_right'
ixname_pairs = [ixname_left, ixname_right]


class EsDfQuery:
    def __init__(self, client, index, doc_type, scoreplan, size=30, explain=True,
                 ixname='ix', lsuffix='left', rsuffix='right',
                 es_id='es_id', es_score='es_score', suffix_score='es', es_rank='es_rank'):
        """

        Args:
            client (elasticsearch.client):
            index (str):
            doc_type (str):
            scoreplan (dict):
            size (int): max number of hits from ES,
            explain (bool): get detailed scores
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            es_score (str): 'es_score'
            es_id (str): 'es_id'
            suffix_score (str): 'es'
            es_rank (str):'es_rank'
        """
        self.client = client
        self.index = index
        self.doc_type = doc_type
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

    def multimatch_esquery(self, record, explain=None, size=None):
        subqueries = list()
        r2 = record.dropna()
        for inputfield in self.scoreplan.keys():
            if inputfield in r2.index:
                scoretype = self.scoreplan[inputfield]['type']
                subquery = dict()
                if scoretype == name_freetext:
                    subquery = {
                        "match": {
                            inputfield: {
                                "query": record[inputfield],
                                "fuzziness": 2,
                                "prefix_length": 2
                            }
                        }
                    }
                elif scoretype == name_exact:
                    subquery = {
                        "match": {
                            inputfield: {
                                "query": record[inputfield]
                            }
                        }
                    }
                else:
                    raise ValueError('scoretype {} differs from [{}, {}]'.format(scoretype, name_exact,
                                                                                 name_freetext))
                subqueries.append(subquery)
        mquery = {
            "size": size,
            "explain": explain,
            "query":
                {
                    "bool": {
                        "should": subqueries
                    }
                }
        }
        return mquery

    def _clientsearch(self, record, explain=None, size=None):
        """

        Args:
            record (pd.Series/dict):

        Returns:
            dict: raw output of ES
        """
        if explain is None:
            explain = self.explain
        if size is None:
            size = self.size
        mquery = self.multimatch_esquery(record, explain=explain, size=size)
        res = self.client.search(
            index=self.index,
            doc_type=self.doc_type,
            body=mquery
        )
        return res

    def search(self, left):
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
            lambda r: self._clientsearch(record=r),
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
        res = self.search(
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
        res (dict):
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


if __name__ == '__main__':
    con_pg = pg.connect("dbname=wookie user=paulogier")
    df = pd.read_sql('SELECT * FROM suppliers_all', con=con_pg).set_index('ix')
    con_pg.close()

    esclient = elasticsearch.Elasticsearch()
    scoreplan = {
        'name': {
            'type': 'FreeText'
        },
        'street': {
            'type': 'FreeText'
        },
        'city': {
            'type': 'FreeText'
        },
        'duns': {
            'type': 'Exact'
        },
        'postalcode': {
            'type': 'FreeText'
        },
        'countrycode': {
            'type': 'Exact'
        }
    }
    escon = EsDfQuery(
        client=esclient,
        scoreplan=scoreplan,
        index="pg-suppliers-all",
        doc_type='supplierid',
        explain=True,
        size=20
    )
    left = df.copy()
    # left_record = df.sample().iloc[0]
    # res = escon._clientsearch(
    #     record=left_record,
    #     explain=True,
    #     size=15
    # )
    # res = unpack_allhits(res, explain=True)
    res = escon.lrsearch(left=df)
    print(res.shape[0] / left.shape[0])
    pass
