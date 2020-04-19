import pandas as pd
from tutorial.main.stepbystep.stepbysteputils.pgconnector import create_engine_ready

engine = create_engine_ready()
# filefolder = '~/'
# leftpath = 'source.csv'
# rightpath = 'target.csv'
# df_source = pd.read_csv(filefolder + leftpath, index_col=0, sep='|', encoding='utf-8')
# df_target = pd.read_csv(filefolder + rightpath, index_col=0, sep='|', encoding='utf-8')

from suricate.data.companies import getsource, gettarget

df_source_raw=getsource(nrows=None)
df_target_raw = gettarget(nrows=None)

def prepare_source(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2

def prepare_target(df):
    """

    Args:
        df:

    Returns:
        pd.DataFrame
    """
    df2 = df
    return df2

df_source = prepare_source(df_source_raw)
df_target = prepare_target(df_target_raw)
assert df_source.columns.equals(df_target.columns)
print(df_source.shape[0])
print(df_target.shape[0])

df_source.to_sql(name='df_source', con=engine, if_exists='replace', index=True)
df_target.to_sql(name='df_target', con=engine, if_exists='replace', index=True)

print('done')