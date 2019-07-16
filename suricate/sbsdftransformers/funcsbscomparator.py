from sklearn.base import TransformerMixin

from suricate.preutils.scores import exact_score, simple_score, token_score, vincenty_score, contain_score
from suricate.sbsdftransformers.base import BaseSbsComparator


class FuncSbsComparator(BaseSbsComparator, TransformerMixin):
    """
    Compare two columns of a dataframe with one another using functions from fuzzywuzzy library
    """

    def __init__(self, on, ixname='ix', lsuffix='left', rsuffix='right', comparator='fuzzy', *args, **kwargs):
        """
        Args:
            comparator (str): name of the comparator function: ['exact', 'fuzzy', 'token', 'contain', 'vincenty' ]
            ixname (str): name of the index, default 'ix'
            lsuffix (str): suffix to be added to the left dataframe default 'left', gives --> 'ix_left'
            rsuffix (str): suffix to be added to the left dataframe default 'right', gives --> 'ixright'
            on (str): name of the column on which to do the join
            *args:
            **kwargs:
        """
        if comparator == 'exact':
            compfunc = exact_score
        elif comparator == 'fuzzy':
            compfunc = simple_score
        elif comparator == 'token':
            compfunc = token_score
        elif comparator == 'vincenty':
            compfunc = vincenty_score
        elif comparator == 'contain':
            compfunc = contain_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token', 'contain', 'vincenty']")
        BaseSbsComparator.__init__(
            self,
            compfunc=compfunc,
            on_left=on + '_' + lsuffix,
            on_right=on + '_' + rsuffix,
            *args,
            **kwargs
        )
        pass


