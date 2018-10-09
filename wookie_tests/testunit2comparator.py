## FOR THE COMPARATOR MODULE
import itertools

import pandas as pd
import pytest


@pytest.mark.skip(reason="Test fine")
def test2(verbose=True):
    # test loading of training and testing data
    trainingpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'

    df_train = pd.read_csv(trainingpath, nrows=500)
    assert isinstance(df_train, pd.DataFrame)
    if verbose is True:
        print('\n length of training_data {}'.format(df_train.shape[0]))
        print('\n columns used {}'.format(df_train.columns.tolist()))
    usecols = ['_'.join(c) for c in itertools.product(
        ['street', 'name'], ['left', 'right'])]
    from wookie import FuzzyWuzzySbsComparator
    left_col = usecols[0]
    right_col = usecols[1]
    fc = FuzzyWuzzySbsComparator(left=left_col, right=right_col, comparator='fuzzy')
    x_2 = fc.fit_transform(df_train)
    print(x_2)
    pass


def test3(verbose=True):
    # test loading of training and testing data
    trainingpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'

    df_train = pd.read_csv(trainingpath, nrows=500)
    usecols = ['_'.join(c) for c in itertools.product(
        ['street', 'name'], ['left', 'right'])]
    from wookie import PipeSbsComparator
    scoreplan = dict()
    for c in ['name', 'street', 'city']:
        scoreplan[c] = ['exact', 'token']

    fc = PipeSbsComparator(scoreplan=scoreplan)
    x_2 = fc.fit_transform(df_train)
    print(x_2)
    pass
