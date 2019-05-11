import numpy as np
import pandas as pd

left = pd.Series(['foo', 'bar', 'baz'])
right = pd.Series(['foo', 'bar', 'geez'])
x = pd.MultiIndex.from_product([left, right], names=['left', 'right'])

df = pd.DataFrame(
    index=x,
    data=np.array(
        [
            np.arange(0, len(x)),
            2 * np.arange(0, len(x))
        ]
    ).transpose()
)


def _getindex(X, y):
    if isinstance(y, pd.MultiIndex):
        return y
    elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        return y.index
    elif y is None:
        ix = pd.MultiIndex.from_product(
            [X[0].index, X[1].index],
            names=['ix_left', 'ix_right']
        )
        return ix
    else:
        print('index, series or dataframe or None expected')
        return y


def fit(X=None, y=None):
    """
    if y is None:
        self.on_ix = X.index --> return X
    else:
        if y is Index
            self.on_ix = y
        else
            if y is Series or DataFrame --> return y.index
    add self.fitted = True
    Args:
        X: default None, X= dataframe of n*m or np.array of n*m
        y: default None

    Returns:
        self
    """
    on_ix = _getindex(X=X, y=y)
    return on_ix


def transform(X, on_ix=None, as_series=False):
    if on_ix is None:
        return X
    else:
        '''
        TODO: Correct
        AttributeError: 'numpy.ndarray' object has no attribute 'index'
        '''
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.loc[on_ix]
        elif isinstance(X, np.ndarray):
            on_ix =
            return X[on_ix]
        else:
            raise TypeError(
                'type of array {} not recognized, should be np.ndarray, pd.DataFrame, pd.Series'.format(type(X)))


n_rows = 3


def test_output(X_fit, X_transform, y, n_rows):
    X_out = transform(X=X_transform, on_ix=fit(X=X_fit, y=y))
    assert X_out.shape[0] == n_rows
    return True


# Case where y is an index, X is a dataframe
test_output(
    X_fit=df,
    X_transform=df,
    y=df.sample(n_rows).index,
    n_rows=n_rows
)

# Case where y is a series, X is a dataframe
test_output(
    X_fit=df,
    X_transform=df,
    y=df.sample(n_rows)[1],
    n_rows=n_rows
)

# Case where y is a series X is a nd array
test_output(
    X_fit=df.values,
    X_transform=df.values,
    y=df.sample(n_rows)[1],
    n_rows=n_rows
)
