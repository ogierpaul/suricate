import numpy as np

from suricate.dftransformers.base import DfTransformerMixin


class ExactConnector(DfTransformerMixin):
    """
    This class returns the cartesian product of source and target, and gives 1 if it is an exact match, 0 otherwise
    """
    def __init__(self, on, ixname='ix', source_suffix='source', target_suffix='target', scoresuffix='exact', **kwargs):
        """

        Args:
            on (str): name of column on which to do the pivot
            ixname (str):
            source_suffix (str):
            target_suffix (str):
            scoresuffix (str): name of score suffix to be added at the end
        Returns
            pd.Series
        """
        DfTransformerMixin.__init__(self, ixname=ixname, source_suffix=source_suffix, target_suffix=target_suffix, on=on,
                                    scoresuffix=scoresuffix, **kwargs)
        pass

    def _transform(self, X):
        """

        Args:
            X (list):

        Returns:
            np.ndarray:  of shape(n_samples_source * n_samples_target, 1)
        """
        ix = self._getindex(X=X)
        ysource = X[0][self.on].values
        ytarget = X[1][self.on].values
        Xcomp = np.transpose([np.repeat(ysource, len(ytarget)), np.tile(ytarget, len(ysource))])
        ynp = np.equal(Xcomp[:, 0], Xcomp[:, 1]).astype(int).reshape(-1, 1)
        return ynp

    def get_feature_names(self):
        """

        Returns:
            list: list of length 1
        """
        return ['_'.join([self.on, 'exact'])]
