import pandas as pd

from wookie.base import BaseSbsComparator
from wookie.comparators.sidebyside import fuzzy_score

nrows = None
sbs = pd.read_csv(
    '/Users/paulogier/81-GithubPackages/wookie/operations/data/trainingdata.csv',
    index_col=[0, 1],
    dtype=str,
    nrows=nrows
)
sbs['y_true'] = sbs['y_true'].astype(int)
scoreplan = {
    'name': ['fuzzy', 'token'],
    'street': ['fuzzy']
}
colname = 'name'
lsuffix = 'left'
rsuffix = 'right'
colnameleft = '_'.join([colname, lsuffix])
colnameright = '_'.join([colname, rsuffix])
comp = BaseSbsComparator(
    on_left=colnameleft,
    on_right=colnameright,
    compfunc=fuzzy_score
)
X_score = comp._ptransform(sbs)
