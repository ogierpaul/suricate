import pandas as pd


def create_foo_database():
    """

    Returns:
        pd.DataFrame, pd.DataFrame
    """
    left = pd.DataFrame(
        {
            'name': ['foo', 'bath'],
            'street': ['paris', 'munich']
        }
    )
    right = pd.DataFrame(
        {
            'name': ['foo', 'bar', 'baz'],
            'street': ['paris', 'munich', 'munich']
        }
    )
    return left, right


def create_gid_database(nrows=100):
    """

    Args:
        nrows (int):

    Returns:
        pd.DataFrame, pd.DataFrame
    """
    leftpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/left_test.csv'
    rightpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/right_test.csv'

    left_data = pd.read_csv(leftpath, index_col=0, nrows=nrows)
    right_data = pd.read_csv(rightpath, index_col=0, nrows=nrows)
    return left_data, right_data


def create_training_database(nrows=500):
    """

    Args:
        nrows (int):

    Returns:
        pd.DataFrame, pd.Series
    """
    trainingpath = '/Users/paulogier/80-PythonProjects/02-test_suricate/data/trainingdata75.csv'

    df_train = pd.read_csv(trainingpath, nrows=nrows)
    y_train = df_train['y_true']
    x_train = df_train[list(filter(lambda c: c != 'y_true', df_train.columns))]
    return x_train, y_train
