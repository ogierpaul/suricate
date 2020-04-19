import pandas as pd

from suricate.dftransformers.base import DfTransformerMixin
from suricate.preutils.similarityscores import exact_score, simple_score, token_score, contains_score, vincenty_score


class DfApplyComparator(DfTransformerMixin):
    def __init__(self, on, ixname='ix', source_suffix='source', target_suffix='target',
                 scoresuffix='fuzzy', compfunc='simple', **kwargs):
        """

        Args:
            on:
            ixname:
            source_suffix:
            target_suffix:
            scoresuffix:
            compfunc (str): ['exact', 'simple', 'token', 'vincenty', 'contain']
            **kwargs:
        """
        DfTransformerMixin.__init__(self, ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix, on=on,
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
            np.array: of shape(n_samples_source * n_samples_target, 1)
        """
        source = X[0]
        target = X[1]
        colnameleft = self.on + '_' + self.source_suffix
        colnameright = self.on + '_' + self.target_suffix
        sbs = pd.DataFrame(
            index=self._getindex(X=X)
        ).reset_index(
            drop=False
        ).join(
            left[self.on], on=self.ixnamesource, how='left'
        ).rename(
            columns={self.on: colnameleft}
        ).join(
            right[self.on], on=self.ixnametarget, how='left'
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
