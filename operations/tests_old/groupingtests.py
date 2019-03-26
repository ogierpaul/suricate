import pandas as pd

from wookie import grouping

filename = 'results_left.csv'
aggmethods = {'name': 'concat',
              'street': 'popularity',
              'city': 'popularity',
              'postalcode': 'popularity',
              'duns': 'last',
              'countrycode': 'popularity'
              }

df_withgid = pd.read_csv(filename, index_col=0, dtype=str)
gids = grouping.calc_goldenrecord(data=df_withgid, gidcol='gid', fieldselector=aggmethods)
print(df_withgid.shape[0], gids.shape[0])
print(gids.head())
gids.to_csv('data/gids_left.csv', index=True, sep=',', encoding='utf-8')
