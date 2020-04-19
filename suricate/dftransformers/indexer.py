from suricate.dftransformers.base import DfTransformerMixin


class Indexer(DfTransformerMixin):
    """
    This class returns a Series with the cartesian product of the index of df_source and df_target
    """
    def __init__(self, ixname='ix', source_suffix='source', target_suffix='target', on=None,
                 scoresuffix=None, **kwargs):
        DfTransformerMixin.__init__(self, ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix, on=on,
                                    scoresuffix=scoresuffix, **kwargs)

    def _transform(self, X):
        """

        Args:
            X (list):

        Returns:
            np.ndarray:  of shape(n_samples_source * n_samples_target, 1)
        """
        ixvals = self._getindex(X=X)
        # ixvals = pd.Series(index=ixvals, data=ixvals.get_values(), name=self.outcol)
        return ixvals.get_values().reshape(-1, 1)
