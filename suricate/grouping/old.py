import pandas as pd

from suricate.preutils.preprocessing import concatenate_names
from suricate.preutils import concatixnames

#TODO: Rework everything

class SingleGrouping:
    def __init__(self,
                 dedupe,
                 data=None,
                 ixname='ix',
                 source_suffix='source',
                 target_suffix='target',
                 gidname='gid',
                 verbose=False):
        """

        Args:
            dedupe (suricate.LrDuplicateFinder):
            data (pd.DataFrame): None
            ixname (str):
            source_suffix (str):
            target_suffix (str):
            gidname (str)
            verbose (bool):
        """
        self.ixname = ixname
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix
        )
        self.dedupe = dedupe
        self.verbose = verbose
        self.gidname = gidname
        if data is not None:
            self.data = data
        else:
            self.data = pd.DataFrame()
        # self.ixnamesource, self.ixnametarget, self.ixnamepairs = concatixnames(
        #     ixname=self.ixname, source_suffix=self.source_suffix, target_suffix=self.target_suffix
        # )

    def launchdedupe(self, data, n_batches=None, n_records=3):
        """

        Args:
            data (pd.DataFrame):
            n_batches (int): Number of batches
            n_records (int): number of records in each batch

        Returns:
            pd.DataFrame
        """
        assert data.index.name == self.ixname
        self.data = data
        # TODO: Initiate for continuous cleansing
        # Init step
        refdata = self._initcleandata()
        i = 0
        while n_batches is None or i < n_batches:
            i += 1
            new = self._choosenextrecord(n_records)
            if new is not None:
                y_proba = self.dedupe.predict(
                    source=new,
                    target=refdata.dropna(subset=[self.gidname]),
                    addmissingsource=True,
                    verbose=True
                )
                self._updategids(y_proba=y_proba)
            else:
                print('{} | deduplication totally finished'.format(pd.datetime.now()))
                return refdata
        print('{} | deduplication finished'.format(pd.datetime.now()))
        return refdata

    def _choosenextrecord(self, n_records=3):
        """

        Args:
            n_records (int): default_3
        Returns:
            pd.DataFrame
        """
        newdata = self.data.loc[self.data[self.gidname].isnull()]
        if newdata.shape[0] == 0:
            return None
        elif newdata.shape[0] >= n_records:
            newrecord = newdata.iloc[:n_records].copy()
        else:
            newrecord = newdata.copy()
        return newrecord

    def _initcleandata(self):
        if self.gidname not in self.data.columns or self.data[self.gidname].dropna().shape[0] == 0:
            self.data[self.gidname] = None
            startix = self.data.index[0]
            y_proba = pd.DataFrame(
                {
                    self.ixnamesource: [startix],
                    self.ixnametarget: [None],
                    'y_proba': [0]
                }
            )
            self._updategids(y_proba=y_proba)
        else:
            pass
        return self.data

    def _updategids(self, y_proba):
        """
        Args:
            y_proba (pd.Series/pd.DataFrame):

        Returns:
            pd.DataFrame
        """
        refdata = self.data.dropna(subset=[self.gidname])
        # results: {rangeix: [ixnamesource, gidname]}
        results = calc_existinggid(
            y_proba=y_proba,
            refdata=refdata,
            ixname=self.ixname,
            source_suffix=self.source_suffix,
            target_suffix=self.target_suffix,
            gidname=self.gidname
        )
        # update refdata
        pos_gids = results.loc[~results.isnull()]
        empty_gids = results.loc[results.isnull()]
        self.data.loc[pos_gids.index, self.gidname] = pos_gids
        self.data.loc[empty_gids.index, self.gidname] = pd.Series(data=empty_gids.index, index=empty_gids.index)
        return self.data


def calc_existinggid(y_proba, refdata, ixname='ix', source_suffix='source', target_suffix='target', gidname='gid'):
    """

    Args:
        y_proba (pd.DataFrame/pd.Series): {[ixnamesource, ixnametarget] : ['y_proba']
        refdata (pd.DataFrame): {ixname:[gidname, cols..]}
        ixname (str):
        source_suffix (str):
        target_suffix (str):
        gidname (str):

    Returns:
        pd.Series: # results :{ ixname: gidname}
    """

    def goodgids(r):
        """
        Return the most common gid
        Args:
            r (pd.Series): {'ixnametarget':'gid'} for a common ixnamesource

        Returns:
            str
        """
        assert isinstance(r, pd.Series)
        vc = r.value_counts()
        if vc.iloc[0] > 1:
            return vc.index[0]
        else:
            return r.iloc[0]

    ixnamesource, ixnametarget, ixnamepairs = concatixnames(
        ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix
    )

    if isinstance(y_proba, pd.Series):
        y_proba = pd.DataFrame(y_proba).reset_index(drop=False)
    for c in ixnamepairs + ['y_proba']:
        assert c in y_proba.columns, '{}'.format(c)
    if isinstance(refdata, pd.Series):
        ref = pd.DataFrame(refdata)
    else:
        assert isinstance(refdata, pd.DataFrame)
        assert gidname in refdata.columns
        assert refdata.index.name == ixname
        ref = refdata[[gidname]].copy()
        ref.index.name = ixnametarget
        ref.reset_index(drop=False, inplace=True)

    # Select positive matches
    pos_matches = y_proba.loc[
        y_proba['y_proba'] > 0.5
        ].copy(
    ).sort_values(
        by='y_proba',
        ascending=False
    )
    # Select left ixes that are NOT in pos matches
    no_matches_atall = y_proba.loc[
        ~(y_proba[ixnamesource].isin(pos_matches[ixnamesource].values)),
        ixnamesource
    ].unique()
    results = pd.DataFrame(data=no_matches_atall, columns=[ixnamesource])
    results[gidname] = None
    # results :{ rangeix: [ixnamesource, gidname]}

    # merge the two to get the the gids
    gids = pd.merge(
        left=pos_matches,
        right=ref,
        left_on=[ixnametarget],
        right_on=[ixnametarget],
        how='inner'
    )
    gb = gids.groupby([ixnamesource])
    wg = pd.DataFrame(gb[gidname].apply(goodgids)).reset_index(drop=False)
    assert isinstance(wg, pd.DataFrame), print(type(wg))
    for c in [ixnamesource, gidname]:
        assert c in wg.columns
    results = pd.concat([results, wg[[ixnamesource, gidname]]], axis=0, ignore_index=True)
    results = results.rename(columns={ixnamesource: ixname}).set_index(ixname)[gidname]
    return results


def calc_goldenrecord(data, gidcol, fieldselector):
    """
    Calculate a golden record
    Args:
        data (pd.DataFrame): {'ix':[cols,..... gid]}
        gidcol (str):
        fieldselector (dict): {colname: 'method'} \
            method in ['popularity', 'first', 'last', 'concat']

    Returns:
        pd.DataFrame
    """
    gb = data.groupby(by=[gidcol])
    df = pd.DataFrame(index=data['gid'].unique())
    for inputfield in fieldselector.keys():
        method = fieldselector[inputfield]
        df[inputfield] = gb[inputfield].apply(lambda r: _agginfo(r, method=method))
    return df


def _agginfo(r, method):
    """

    Args:
        r:
        method:

    Returns:
        scalar
    """
    possiblemethods = ['popularity', 'first', 'last', 'concat']
    r2 = _checkvalue(r)
    if r2 is None:
        return None
    else:
        if method == 'popularity':
            return _popularity(r2, keep='first')
        elif method in ['first', 'last']:
            return _byorder(r2, keep=method)
        elif method == 'concat':
            return _smartconcat(r2)
        else:
            raise ValueError('method {} not in possiblemethods {}'.format(method, possiblemethods))


def _checkvalue(r):
    r2 = r.dropna()
    if r2.shape[0] == 0:
        return None
    else:
        return r2


def _popularity(r, keep='first'):
    """
    return the most common value for this series
    if there are none: return
    i
    Args:
        r (pd.Series):
        keep (str):

    Returns:
        scalar
    """

    vc = r.value_counts()
    if vc.iloc[0] > 1:
        return vc.index[0]
    else:
        return _byorder(r, keep=keep)


def _byorder(r, keep='last'):
    """
    Args:
        r (pd.Series):
        keep (str):

    Returns:
        scalar
    """
    r2 = r.dropna()
    if keep == 'first':
        return r2.iloc[0]
    elif keep == 'last':
        return r2.iloc[-1]


def _smartconcat(r):
    """

    Args:
        r:

    Returns:
        aggregate view
    """
    return concatenate_names(r)
