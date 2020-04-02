import pandas as pd
from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready

engine = create_engine_ready()
# filefolder = '~/'
# leftpath = 'left.csv'
# rightpath = 'right.csv'
# df_left = pd.read_csv(filefolder + leftpath, index_col=0, sep='|', encoding='utf-8')
# df_right = pd.read_csv(filefolder + rightpath, index_col=0, sep='|', encoding='utf-8')

from suricate.data.companies import getleft, getright

df_left_raw=getleft(nrows=None)
df_right_raw = getright(nrows=None)

def prepare_left(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2

def prepare_right(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2

df_left = prepare_left(df_left_raw)
df_right = prepare_right(df_right_raw)
assert df_left.columns.equals(df_right.columns)
print(df_left.shape[0])
print(df_right.shape[0])

df_left.to_sql(name='df_left', con=engine, if_exists='replace', index=True)
df_right.to_sql(name='df_right', con=engine, if_exists='replace', index=True)

print('done')