from suricate.lrdftransformers.base import LrDfTransformerMixin


class Indexer(LrDfTransformerMixin):
    """
    This class returns a Series with the cartesian product of the index of df_left and df_right
    """
    def __init__(self, ixname='ix', lsuffix='left', rsuffix='right', on=None,
                 scoresuffix=None, **kwargs):
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix, **kwargs)

    def _transform(self, X):
        """

        Args:
            X (list):

        Returns:
            np.ndarray:  of shape(n_samples_left * n_samples_right, 1)
        """
        ixvals = self._getindex(X=X)
        # ixvals = pd.Series(index=ixvals, data=ixvals.get_values(), name=self.outcol)
        return ixvals.get_values().reshape(-1, 1)
