import numpy as np

from wookie.lrdftransformers.base import LrDfTransformerMixin


class ExactConnector(LrDfTransformerMixin):
    '''
    This class returns the cartesian product
    '''
    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right', scoresuffix='exact', **kwargs):
        """

        Args:
            on (str): name of column on which to do the pivot
            ixname (str):
            lsuffix (str):
            rsuffix (str):
            scoresuffix (str): name of score suffix to be added at the end
        Returns
            pd.Series
        """
        LrDfTransformerMixin.__init__(self, ixname=ixname, lsuffix=lsuffix, rsuffix=rsuffix, on=on,
                                      scoresuffix=scoresuffix, **kwargs)
        pass

    def _transform(self, X):
        """

        Args:
            X (list):

        Returns:
            np.ndarray:  of shape(n_samples_left * n_samples_right, 1)
        """
        ix = self._getindex(X=X)
        yleft = X[0][self.on].values
        yright = X[1][self.on].values
        Xcomp = np.transpose([np.repeat(yleft, len(yright)), np.tile(yright, len(yleft))])
        ynp = np.equal(Xcomp[:, 0], Xcomp[:, 1]).astype(int).reshape(-1, 1)
        return ynp
