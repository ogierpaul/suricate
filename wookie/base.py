# import pandas as pd
import pandas as pd
from sklearn.base import TransformerMixin


# This dont do anything, just pass the data as it is
class DataPasser(TransformerMixin):
    """
    This dont do anything, just pass the data as it is
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        This dont do anything, just pass the data as it is
        Args:
            X:

        Returns:

        """
        return X


class BaseComparator(TransformerMixin):
    def __init__(self, left='left', right='right', compfunc=None, *args, **kwargs):
        """
        base class for all transformers
        Args:
            left (str):
            right (str):
            compfunc (function): ['fuzzy', 'token', 'exact']
        """
        TransformerMixin.__init__(self)
        self.left = left
        self.right = right
        if compfunc is None:
            raise ValueError('comparison function not provided with function', compfunc)
        assert callable(compfunc)
        self.compfunc = compfunc

    def transform(self, X):
        """
        Args:
            X (pd.DataFrame):

        Returns:
            np.ndarray
        """
        compfunc = self.compfunc
        if not compfunc is None:
            y = X.apply(
                lambda r: compfunc(
                    r.loc[self.left],
                    r.loc[self.right]
                ),
                axis=1
            ).values.reshape(-1, 1)
            return y
        else:
            raise ValueError('compfunc is not defined')

    def fit(self, *_):
        return self


class BaseConnector(TransformerMixin):
    """
    Class inherited from TransformerMixin
    Attributes:
        attributesCols (list): list of column names desribing the data
        relevanceCol (str): name of column describing how relevant the data is to the query
        left_index (str): suffix to identify columns from the left dataset
        right_index (str): suffix to identify columns from the right dataset
    """

    def __init__(self, *args, **kwargs):
        TransformerMixin.__init__(self)
        self.attributesCols = []
        self.relevanceCol = []
        self.left_index = 'left_index'
        self.right_index = 'right_index'
        pass

    def transform(self, X, *_):
        """

        Args:
            X (pd.DataFrame): array containing the query
            *_:

        Returns:
            pd.DataFrame
        """
        assert isinstance(X, pd.DataFrame)
        X['relevance_score'] = None
        result = X
        return result

    def fit(self, *_):
        return self
