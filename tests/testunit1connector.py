import pandas as pd

## FOR THE CONNECTOR MODULE


def test2(verbose=True):
    # test loading of training and testing data
    # trainingpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'
    leftpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/left_test.csv'
    rightpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/right_test.csv'

    left_data = pd.read_csv(leftpath, index_col=0, nrows=100)
    right_data = pd.read_csv(rightpath, index_col=0, nrows=100)
    assert isinstance(left_data, pd.DataFrame)
    if verbose is True:
        print('\n length of left_data {}'.format(left_data.shape[0]))
        print('\n length of right_data {}'.format(right_data.shape[0]))
        print('\n columns used {}'.format(left_data.columns.tolist()))
    return left_data, right_data


def test3():
    # test implementation of transformer
    left_data, right_data = test2(verbose=False)

    from sklearn.base import TransformerMixin
    from wookie.connectors import Cartesian
    import itertools

    con = Cartesian(reference=right_data)
    assert isinstance(con, TransformerMixin)
    df2 = con.fit(left_data).transform(left_data)

    print(df2.shape[0] == (left_data.shape[0] * right_data.shape[0]))
    print(df2.columns)
    print(df2.shape[0])
    print(df2['relevance_score'].max())
    displaycols = ['name']
    usecols = ['_'.join(c) for c in itertools.product(displaycols, ['left', 'right'])]
    print(df2.sort_values(by=['relevance_score'], ascending=False).loc[:, usecols].head())
    pass
