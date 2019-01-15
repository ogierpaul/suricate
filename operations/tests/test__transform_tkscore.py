from unittest import TestCase

import pandas as pd

from wookie.comparators.leftright.tokenizers import LrTokenComparator
from wookie.connectors.dataframes.base import cartesian_join

left = pd.read_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/left.csv', index_col=0)
right = pd.read_csv('/Users/paulogier/81-GithubPackages/wookie/operations/data/right.csv', index_col=0)
comparator = LrTokenComparator(
    vectorizermodel='tfidf',
    ngram_range=(2, 2),
    analyzer='char'
)


class Test_transform_tkscore(TestCase):
    tf_score = comparator.fit(left=left['name'], right=right['name']).transform(left=left, right=right)
    sbs = cartesian_join(left=left, right=right)

    assert isinstance(score, list)
    pass
