"""
Test mxutils functions
"""

import os

import pytest

from nephosem import TypeTokenMatrix
from nephosem.specutils import mxutils

curdir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def matrices():
    mxdict1 = {
        'a': {'b': 1, 'c': 2},
        'b': {'c': 3, 'e': 2},
        'c': {'d': 4, 'e': 1},
    }
    mxdict2 = {
        'b': {'c': 1, 'e': 2},
        'c': {'d': 3, 'e': 2},
        'f': {'c': 4, 'd': 1},
    }
    row1, col1 = ['a', 'b', 'c'], ['b', 'c', 'd', 'e']
    row2, col2 = ['b', 'c', 'f'], ['c', 'd', 'e']
    mx1 = mxutils.transform_dict_to_spmatrix(mxdict1, ['a', 'b', 'c'], ['b', 'c', 'd', 'e'])
    mx2 = mxutils.transform_dict_to_spmatrix(mxdict2, ['b', 'c', 'f'], ['c', 'd', 'e'])
    mx1 = TypeTokenMatrix(mx1, row1, col1)
    mx2 = TypeTokenMatrix(mx2, row2, col2)
    yield [mx1, mx2]


class TestMxUtils(object):
    def test_merge_matrix_dict(self, matrices):
        mxdict_list = []
        for mx in matrices:
            mxd = mxutils.transform_spmatrix_to_dict(mx.matrix, mx.row_items, mx.col_items)
            mxdict_list.append(mxd)
        res_dict = mxutils.merge_matrix_dict(mxdict_list)
        print(res_dict)
        assert False

    def test_merge_matrices(self, matrices):
        resmx, resrow, rescol = mxutils.merge_matrices(matrices)
        print(TypeTokenMatrix(resmx, resrow, rescol))
        assert False
