## FOR THE COMPARATOR MODULE
import itertools

import pandas as pd

from wookie.estimators import RandomForestClassifier
from wookie.utils import exact_score


def test1():
    # test import of comparator module
    from wookie.utils import navalue_score
    assert isinstance(navalue_score, float)
    pass


def test2(verbose=True):
    # test loading of training and testing data
    trainingpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'

    df_train = pd.read_csv(trainingpath, nrows=500)
    assert isinstance(df_train, pd.DataFrame)
    if verbose is True:
        print('\n length of training_data {}'.format(df_train.shape[0]))
        print('\n columns used {}'.format(df_train.columns.tolist()))
    usecols = ['_'.join(c) for c in itertools.product(['street'], ['left', 'right'])]
    df2 = df_train.loc[:, usecols]
    from wookie.comparators import BaseComparator, FuzzyWuzzyComparator
    bc = BaseComparator(compfunc=exact_score, left=usecols[0], right=usecols[1], outputCol='out_col')
    df2 = bc.transform(df2)
    fc = FuzzyWuzzyComparator(left=usecols[0], right=usecols[1], outputCol='out_col', comparator='fuzzy')
    df2 = fc.transform(df2)
    stages = []
    for c in ['name', 'street', 'city']:
        left = '_'.join([c, 'left'])
        right = '_'.join([c, 'right'])
        for comp in ['exact', 'fuzzy', 'token']:
            outputCol = '_'.join([comp, c])
            stages.append(
                FuzzyWuzzyComparator(left=left, right=right, comparator=comp, outputCol=outputCol)
            )
    outputCols = [c.outputCol for c in stages]
    X = df_train.copy()
    for c in stages:
        X = c.transform(X)
    classifier = RandomForestClassifier()
    classifier.fit(X.loc[:, outputCols], X['y_true'])
    score = classifier.score(X.loc[:, outputCols], X['y_true'])
    print('classifier score', score)
    pass
