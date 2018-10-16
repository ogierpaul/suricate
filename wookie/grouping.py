import pandas as pd


class SingleGrouping:
    def __init__(self,
                 dedupe,
                 data=None,
                 ixname='ix',
                 lsuffix='_left',
                 rsuffix='_right',
                 gidname='gid',
                 verbose=False):
        """

        Args:
            dedupe (wookie.LrDuplicateFinder):
            data (pd.DataFrame): None
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            gidname (str)
            verbose (bool):
        """
        self.ixname = ixname
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.dedupe = dedupe
        self.verbose = verbose
        self.gidname = gidname
        if not data is None:
            self.data = data
        else:
            self.data = pd.DataFrame()
        self._ixnameleft = self.ixname + self.lsuffix
        self._ixnameright = self.ixname + self.rsuffix
        self._ixnamepairs = [self._ixnameleft, self._ixnameright]

    def findduplicates(self, data, n_batches=None, n_records=3):
        """

        Args:
            data (pd.DataFrame):

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
                    left=new,
                    right=refdata.dropna(subset=[self.gidname]),
                    addmissingleft=True,
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
        if not self.gidname in self.data.columns or self.data[self.gidname].dropna().shape[0] == 0:
            self.data[self.gidname] = None
            startix = self.data.index[0]
            y_proba = pd.DataFrame(
                {
                    self._ixnameleft: [startix],
                    self._ixnameright: [None],
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
        # results: {rangeix: [ixnameleft, gidname]}
        results = calc_existinggid(
            y_proba=y_proba,
            refdata=refdata,
            ixnameleft=self._ixnameleft,
            ixnameright=self._ixnameright,
            ixname='ix',
            gidname=self.gidname
        )
        # update refdata
        pos_gids = results.loc[~results.isnull()]
        empty_gids = results.loc[results.isnull()]
        self.data.loc[pos_gids.index, self.gidname] = pos_gids
        self.data.loc[empty_gids.index, self.gidname] = pd.Series(data=empty_gids.index, index=empty_gids.index)
        return self.data


def calc_existinggid(y_proba, refdata, ixnameleft='ix_left', ixnameright='ix_right', ixname='ix', gidname='gid'):
    """

    Args:
        y_proba (pd.DataFrame/pd.Series): {[ixnameleft, ixnameright] : ['y_proba']
        refdata (pd.DataFrame): {ixname:[gidname, cols..]}
        ixnameleft (str):
        ixnameright (str):
        ixname (str):
        gidname (str):

    Returns:
        pd.Series: # results :{ ixname: gidname}
    """

    def goodgids(r):
        assert isinstance(r, pd.Series)
        vc = r.value_counts()
        if vc.iloc[0] > 1:
            return vc.index[0]
        else:
            return r.iloc[0]

    ixnamepairs = [ixnameleft, ixnameright]
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
        ref.index.name = ixnameright
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
        ~(y_proba[ixnameleft].isin(pos_matches[ixnameleft].values)),
        ixnameleft
    ].unique()
    results = pd.DataFrame(data=no_matches_atall, columns=[ixnameleft])
    results[gidname] = None
    # results :{ rangeix: [ixnameleft, gidname]}

    gids = pd.merge(
        left=pos_matches,
        right=ref,
        left_on=[ixnameright],
        right_on=[ixnameright],
        how='inner'
    )
    gb = gids.groupby([ixnameleft])
    wg = pd.DataFrame(gb[gidname].apply(goodgids)).reset_index(drop=False)
    assert isinstance(wg, pd.DataFrame), print(type(wg))
    for c in [ixnameleft, gidname]:
        assert c in wg.columns
    results = pd.concat([results, wg[[ixnameleft, gidname]]], axis=0, ignore_index=True)
    results = results.rename(columns={ixnameleft: ixname}).set_index(ixname)[gidname]
    return results


def calc_goldenrecord(data, gidcol, fieldselector):
    """

    Args:
        data:
        gidcol:
        fieldselector:

    Returns:

    """
