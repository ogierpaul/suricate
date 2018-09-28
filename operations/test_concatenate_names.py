from unittest import TestCase

from wookie.companypreparation import concatenate_names


class TestConcatenate_names(TestCase):
    def test1(self):
        names = ['batman', 'batman aka frank knight']
        res = concatenate_names(names)
        print('\n, ', res)
        assert isinstance(res, str)
