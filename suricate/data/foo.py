import pandas as pd
from suricate.data.base import ix_names
from suricate.lrdftransformers import cartesian_join
from suricate.preutils.indextools import createmultiindex
_samplecol = 'name'


left = pd.DataFrame(
    {
        ix_names['ixname']: [0, 1, 2],
        _samplecol: ['foo', 'bar', 'ninja']
    }
).set_index(
    ix_names['ixname']
)

right = pd.DataFrame(
    {
        ix_names['ixname']: [0, 1, 2],
        _samplecol: ['foo', 'bar', 'baz']
    }
).set_index(
    ix_names['ixname']
)

X_lr = [left, right]

X_sbs = cartesian_join(
    left=X_lr[0],
    right=X_lr[1],
    lsuffix=ix_names['lsuffix'],
    rsuffix=ix_names['rsuffix'],
).set_index(
    ix_names['ixnamepairs']
)

y_true = pd.Series(
    data=[1, 0, 0, 0, 1, 1, 0, 0, 0],
    index=createmultiindex(
        X=X_lr,
        names=ix_names['ixnamepairs']
    ),
    name='y_true'
)