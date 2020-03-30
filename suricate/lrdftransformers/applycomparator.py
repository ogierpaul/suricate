import pandas as pd

from suricate.lrdftransformers.base import LrDfTransformerMixin
from suricate.preutils.scores import exact_score, simple_score, token_score, contains_score, vincenty_score


class LrApplyComparator(LrDfTransformerMixin):
    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right',
                 scoresuffix='fuzzy', compfunc='simple', **kwargs):
        """

        Args:
            on:
            ixname:
            lsuffix:
            rsuffix:
            scoresuffix:
            compfunc (str): ['exact', 'simple', 'token', 'vincenty', 'contain']
            **kwargs:
        """
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix + '_' + compfunc, **kwargs)
        self.on = on
        assert compfunc in ['exact', 'simple', 'token', 'vincenty', 'contain']
        if compfunc == 'simple':
            self.func = simple_score
        elif compfunc == 'token':
            self.func = token_score
        elif compfunc == 'exact':
            self.func = exact_score
        elif compfunc == 'vincenty':
            self.func = vincenty_score
        elif compfunc == 'contain':
            self.func = contains_score

    def _transform(self, X):
        """

        Args:
            X:

        Returns:
            np.array: of shape(n_samples_left * n_samples_right, 1)
        """
        left = X[0]
        right = X[1]
        colnameleft = self.on + '_' + self.lsuffix
        colnameright = self.on + '_' + self.rsuffix
        sbs = pd.DataFrame(
            index=self._getindex(X=X)
        ).reset_index(
            drop=False
        ).join(
            left[self.on], on=self.ixnameleft, how='left'
        ).rename(
            columns={self.on: colnameleft}
        ).join(
            right[self.on], on=self.ixnameright, how='left'
        ).rename(
            columns={self.on: colnameright}
        ).set_index(
            self.ixnamepairs
        )
        sbs[self.outcol] = sbs.apply(
            lambda r: self.func(r[colnameleft], r[colnameright]),
            axis=1
        )
        score = sbs[self.outcol]
        return score.values.reshape(-1, 1)
