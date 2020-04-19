from sklearn.base import TransformerMixin

from suricate.preutils.similarityscores import exact_score, simple_score, token_score, vincenty_score, contains_score
from suricate.sbsdftransformers.base import BaseSbsComparator


class FuncSbsComparator(BaseSbsComparator, TransformerMixin):
    """
    Compare two columns of a dataframe with one another using functions from fuzzywuzzy library
    """

    def __init__(self, on, ixname='ix', source_suffix='source', target_suffix='target', comparator='fuzzy', *args, **kwargs):
        """
        Args:
            comparator (str): name of the comparator function: ['exact', 'fuzzy', 'token', 'contains', 'vincenty' ]
            ixname (str): name of the index, default 'ix'
            source_suffix (str): suffix to be added to the left dataframe default 'left', gives --> 'ix_source'
            target_suffix (str): suffix to be added to the left dataframe default 'right', gives --> 'ixright'
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
        elif comparator == 'contains':
            compfunc = contains_score
        else:
            raise ValueError('compfunc value not understood: {}'.format(comparator),
                             "must be one of those: ['exact', 'fuzzy', 'token', 'contains', 'vincenty']")
        BaseSbsComparator.__init__(
            self,
            compfunc=compfunc,
            on_source=on + '_' + source_suffix,
            on_target=on + '_' + target_suffix,
            *args,
            **kwargs
        )
        pass


