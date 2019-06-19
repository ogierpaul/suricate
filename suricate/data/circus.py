import pandas as pd
import pytest

from suricate.lrdftransformers import CartesianDataPasser
from suricate.preutils.indextools import createmultiindex

left = pd.DataFrame(
    {
        'name': [
            'hello world',
            'hello big world',
            'holy grail',
            'holy moly',
            None,
            'HELLO! world'
        ]
    }
)
left.index.name = 'ix'

right = left.copy()
right['ix'] = pd.Series(list('abcdef'), index=right.index)
right.set_index('ix', drop=True, inplace=True)

X_lr = [left, right]

X_sbs = CartesianDataPasser(
    ixname='ix', lsuffix='left', rsuffix='right', on='name'
).transform(
    X=X_lr
).set_index(
    ['ix_left', 'ix_right']
)

y_true = pd.Series(
    index=createmultiindex(X=X_lr),
    data=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    name='y_true'
)
if __name__ == '__main__':
    print(X_sbs)
