import pandas as pd

from suricate.lrdftransformers.base import LrDfTransformerMixin
from suricate.preutils.scores import exact_score, simple_score, token_score, contain_score, vincenty_score


class FuzzyConnector(LrDfTransformerMixin):
    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right',
                 scoresuffix='fuzzy', ratio='simple', **kwargs):
        """

        Args:
            on:
            ixname:
            lsuffix:
            rsuffix:
            scoresuffix:
            ratio (str): ['exact', 'simple', 'token', 'vincenty', 'contain']
            **kwargs:
        """
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix + '_' + ratio, **kwargs)
        self.on = on
        assert ratio in ['exact', 'simple', 'token', 'vincenty', 'contain']
        if ratio == 'simple':
            self.func = simple_score
        elif ratio == 'token':
            self.func = token_score
        elif ratio == 'exact':
            self.func = exact_score
        elif ratio == 'vincenty':
            self.func = vincenty_score
        elif ratio == 'contain':
            self.func = contain_score

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
