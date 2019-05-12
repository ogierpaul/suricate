import pandas as pd
from wookie.comparators.fuzzy import token_score, simple_score

from wookie.lrdftransformers.base import LrDfTransformerMixin


# FOR FUTURE STEPS
# ENRICH THE NUMBER OF FUNCTIONS
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
            ratio (str): ['simple' , 'token']
            **kwargs:
        """
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix + '_' + ratio, **kwargs)
        self.on = on
        assert ratio in ['simple', 'token']
        if ratio == 'simple':
            self.func = simple_score
        else:
            self.func = token_score

    def _transform(self, X, on_ix=None):
        left = X[0]
        right = X[1]
        colnameleft = self.on + '_' + self.lsuffix
        colnameright = self.on + '_' + self.rsuffix
        sbs = pd.DataFrame(
            index=self._getindex(X=X, y=on_ix)
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
        return score
