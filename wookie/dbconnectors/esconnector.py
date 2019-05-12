from multiprocessing import Pool

import elasticsearch
import numpy as np
import pandas as pd
import psycopg2 as pg
from wookie.lrcomparators import _namescoreplan_freetext, _namescoreplan_id

from wookie.dbconnectors import addsuffix
from wookie.preutils import _ixnames


def parallelize(data, func, n_jobs=2):
    data_split = np.array_split(data, n_jobs)
    pool = Pool(n_jobs)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


class EsDfQuery:
    def __init__(self, client, index, doc_type, scoreplan, max_hits=30,
                 ixname='ix', lsuffix='left', rsuffix='right', scorecol='es_rel'):
        """

        Args:
            client (elasticsearch.client):
            index (str):
            doc_type (str):
            scoreplan (dict):
            max_hits (int): max number of hits from ES
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            scorecol (str)
        """
        self.client = client
        self.index = index
        self.doc_type = doc_type
        self.scoreplan = scoreplan
        self.max_hits = max_hits
        self.scorecol = scorecol
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.ixnameleft, self.ixnameright, self.ixnamepairs = _ixnames(
            ixname=ixname,
            lsuffix=lsuffix,
            rsuffix=rsuffix
        )
        self.usecols = list(self.scoreplan.keys())

    def multimatch_esquery(self, record):
        subqueries = list()
        for inputfield in self.scoreplan.keys():
            if pd.isnull(record[inputfield]) is False:
                scoretype = self.scoreplan[inputfield]['type']
                subquery = dict()
                if scoretype == _namescoreplan_freetext:
                    subquery = {
                        "match": {
                            inputfield: {
                                "query": record[inputfield],
                                "fuzziness": 2,
                                "prefix_length": 1
                            }
                        }
                    }
                elif scoretype == _namescoreplan_id:
                    subquery = {
                        "match": {
                            inputfield: {
                                "query": record[inputfield]
                            }
                        }
                    }
                else:
                    raise ValueError('scoretype {} differs from [{}, {}]'.format(scoretype, _namescoreplan_id,
                                                                                 _namescoreplan_freetext))
                subqueries.append(subquery)
        mquery = {
            "size": self.max_hits,
            "query":
                {
                    "bool": {
                        "should": subqueries
                    }
                }
        }
        return mquery

    def _clientsearch(self, record):
        """

        Args:
            record (pd.Series/dict):

        Returns:
            dict
        """
        mquery = self.multimatch_esquery(record)
        res = self.client.search(
            index=self.index,
            doc_type=self.doc_type,
            body=mquery
        )
        res = unpack_hits(res=res, scorecol=self.scorecol)
        return res

    def search(self, record):
        """

        Args:
            record (dict/pd.Series):

        Returns:
            pd.DataFrame
        """

        res = self._clientsearch(record=record)
        res = pd.DataFrame.from_dict(
            res,
            orient='index'
        ).sort_values(
            by=self.scorecol,
            ascending=False
        )
        res[self.ixnameleft] = record.name
        res.rename({self.ixname: self.ixnameright}, axis=1, inplace=True)
        res.set_index(self.ixnamepairs, inplace=True)
        return res

    def psearch(self, left):
        """
        Args:
            left (pd.DataFrame):
        Returns:
            pd.DataFrame: {['ix_left', 'ix_right'] : [relevance_score, name, ...]}
        """
        tempcol1 = 'f1b3'
        tempcol2 = 'f2b4'
        res = left.apply(self._clientsearch, axis=1)
        fres = pd.melt(
            res.apply(
                pd.Series
            ).reset_index(
                drop=False
            ).rename(
                {self.ixname: self.ixnameleft},
                axis=1
            ),
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
        fres2.set_index(self.ixnamepairs, inplace=True)
        return fres2

    def lrsearch(self, left, usecols='all'):
        """

        Args:
            left (pd.DataFrame):
            usecols (str): in ['all': only use use_cols

        Returns:
            pd.DataFrame
        """
        res = self.psearch(
            left=left
        )
        sbs = res.reset_index(drop=False).loc[:, self.ixnamepairs + self.usecols + [self.scorecol]]
        renamecols = list(
            filter(
                lambda x: x not in [self.ixnameleft, self.ixnameright, self.scorecol],
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


def unpack_hits(res, scorecol='score'):
    """

    Args:
        res (dict):
        scorecol (str):

    Returns:
        list
    """
    sourcedata = res['hits']['hits']
    results = []
    for c in sourcedata:
        mydict = dict()
        mydict['es_id'] = c['_id']
        mydict[scorecol] = c['_score']
        mydict.update(c['_source'])
        results.append(mydict)
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
        doc_type='supplierid'
    )
    left_record = df.sample().iloc[0]
    res = escon.lrsearch(left=df.sample(5))
    print(res.index.names)
    print(res.columns.tolist())
