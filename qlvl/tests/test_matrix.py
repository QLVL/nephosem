"""
Test TypeTokenMatrix Class
"""

import os
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import squareform

from qlvl import TypeTokenMatrix

curdir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def spMTX():
    # [-5,  0, -3, -2]
    # [-1,  0,  0, +1]
    # [+2,  0, +4, +5]
    # this is a sparse matrix (scipy.sparse.csr_matrix)
    sparr = np.array([
        [-5, 0, -3, -2],
        [-1, 0, 0, 1],
        [2, 0, 4, 5]
    ])
    spmx = sp.csr_matrix(sparr)
    spmx[1, 3] = 0  # make the value -> explicit zero
    row_items = ['row0', 'row1', 'row2']
    col_items = ['col0', 'col1', 'col2', 'col3']
    spMTX = TypeTokenMatrix(spmx, row_items, col_items)
    yield spMTX


@pytest.fixture()
def nmMTX():
    # [-5, -4, -3, -2]
    # [-1,  0,  0, +1]
    # [+2, +3, +4, +5]
    # this is a dense matrix (numpy.ndarray)
    arr = np.array([
        [-5, -4, -3, -2],
        [-1, 0, 0, 1],
        [2, 3, 4, 5]
    ])
    dsmx = arr
    row_items = ['row0', 'row1', 'row2']
    col_items = ['col0', 'col1', 'col2', 'col3']
    nmMTX = TypeTokenMatrix(dsmx, row_items, col_items)
    yield nmMTX


@pytest.fixture()
def sqMTX():
    # this is a symmetric dense matrix (numpy.ndarray)
    sqarr = np.array([1, 2, 3, 4, 5, 6])
    sqmx = squareform(sqarr)
    col_items = ['col0', 'col1', 'col2', 'col3']
    sqMTX = TypeTokenMatrix(sqmx, col_items, col_items)
    yield sqMTX


class TestTypeTokenMatrix(object):
    def getitem(self, spMTX, nmMTX, sqMTX):
        # test __getitem__ method for sparse matrix (scipy.sparse.csr_matrix)
        row1 = np.array([-1, 0, 0, 0])
        assert np.array_equal(spMTX['row1'], row1)
        assert np.array_equal(spMTX[1], row1)
        assert np.array_equal(spMTX.matrix[1].toarray()[0], row1)

        col2 = np.array([[-3], [0], [4]])
        assert np.array_equal(spMTX[:, 'col2'], col2)
        assert np.array_equal(spMTX[:, 2], col2)
        assert np.array_equal(spMTX.matrix[:, 2].toarray(), col2)

        val02 = 0
        assert spMTX['row1', 'col2'] == val02
        assert spMTX[1, 2] == val02

        # slicing examples
        assert isinstance(spMTX[:2], TypeTokenMatrix)
        assert isinstance(spMTX[:2, :2], TypeTokenMatrix)
        assert np.array_equal(spMTX[:2][:, :2], spMTX[:2, :2])

        # test __getitem__ method for normal matrix (numpy.ndarray)
        row1 = np.array([-1, 0, 0, 1])
        assert np.array_equal(spMTX['row1'], row1)
        assert np.array_equal(spMTX[1], row1)
        assert np.array_equal(spMTX.matrix[1].toarray, row1)

        col2 = np.array([[-3], [0], [4]])
        assert np.array_equal(spMTX[:, 'col2'], col2)
        assert np.array_equal(spMTX[:, 2], col2)
        assert np.array_equal(spMTX.matrix[:, 2].toarray(), col2)

        val02 = -3
        assert spMTX['row1', 'col2'] == val02
        assert spMTX[1, 2] == val02

        # slicing examples
        assert isinstance(spMTX[:2], TypeTokenMatrix)
        assert isinstance(spMTX[:2, :2], TypeTokenMatrix)
        assert np.array_equal(spMTX[:2][:, :2], spMTX[:2, :2])

    def test_submatrix(self, spMTX, nmMTX, sqMTX):
        # test sparse
        subrow = ['row0', 'row1']
        subcol = ['col1', 'col2']

        subspMTX = spMTX.submatrix(row=subrow)
        # the following slicing would keep the explicit zeros
        # subspMTX2 = TypeTokenMatrix(spMTX.matrix[:2], subrow, spMTX.col_items)
        subspMTX2 = TypeTokenMatrix(spMTX.matrix[[0, 1]], subrow, spMTX.col_items)
        assert subspMTX.equal(subspMTX2)
        subspMTX = spMTX.submatrix(col=subcol)
        subspMTX2 = TypeTokenMatrix(spMTX.matrix[:, [1, 2]], spMTX.row_items, subcol)
        assert subspMTX.equal(subspMTX2)
        subspMTX = spMTX.submatrix(row=['row0', 'row1'], col=['col1', 'col2'])
        submx = spMTX.matrix[[0, 1]].tocsc()
        submx = submx[:, [1, 2]].tocsr()
        subspMTX2 = TypeTokenMatrix(submx, subrow, subcol)
        assert subspMTX.equal(subspMTX2)

        # test normal
        subnmMTX = nmMTX.submatrix(row=subrow)
        subnmMTX2 = TypeTokenMatrix(nmMTX.matrix[[0, 1]], subrow, nmMTX.col_items)
        assert subnmMTX.equal(subnmMTX2)
        subnmMTX = nmMTX.submatrix(col=subcol)
        subnmMTX2 = TypeTokenMatrix(nmMTX.matrix[:, [1, 2]], nmMTX.row_items, subcol)
        assert subnmMTX.equal(subnmMTX2)
        subnmMTX = nmMTX.submatrix(row=subrow, col=subcol)
        submx = nmMTX.matrix[[0, 1]]
        submx = submx[:, [1, 2]]
        subnmMTX2 = TypeTokenMatrix(submx, subrow, subcol)
        assert subnmMTX.equal(subnmMTX2)

        # test square
        subsqMTX = sqMTX.submatrix(row=subcol, col=subcol)
        submx = sqMTX.matrix[[1, 2]]
        submx = submx[:, [1, 2]]
        subsqMTX2 = TypeTokenMatrix(submx, subcol, subcol)
        assert subsqMTX.equal(subsqMTX2)

    def test_sample(self, spMTX, nmMTX, sqMTX):
        samplespMTX = spMTX.sample(percent=0.5)
        print(samplespMTX)
        assert isinstance(samplespMTX, TypeTokenMatrix)
        samplenmMTX = nmMTX.sample(percent=0.5)
        print(samplenmMTX)
        assert isinstance(samplespMTX, TypeTokenMatrix)
        # sqMTX does not have this method

    def test_reorder(self, spMTX, nmMTX, sqMTX):
        new_row_items = ['row1', 'row2', 'row0']
        new_col_items = ['col2', 'col0', 'col3', 'col1']

        reospMTX = spMTX.reorder(new_row_items)
        assert reospMTX.row_items == new_row_items
        reospMTX = spMTX.reorder(new_col_items, axis=1)
        assert reospMTX.col_items == new_col_items

        reonmMTX = nmMTX.reorder(new_row_items)
        assert reonmMTX.row_items == new_row_items
        reonmMTX = nmMTX.reorder(new_col_items, axis=1)
        assert reonmMTX.col_items == new_col_items

    def test_operators(self, spMTX, nmMTX, sqMTX):
        # test sparse
        resMTX = spMTX.multiply(spMTX > 2)
        assert resMTX.matrix.nnz == 2
        resMTX = spMTX.multiply(spMTX >= 2)
        assert resMTX.matrix.nnz == 3
        resMTX = spMTX.multiply(spMTX < 2)
        assert resMTX.matrix.nnz == 5
        resMTX = spMTX.multiply(spMTX <= 2)
        assert resMTX.matrix.nnz == 6
        resMTX = spMTX.multiply(spMTX > -1)
        assert resMTX.matrix.nnz == 4
        resMTX = spMTX.multiply(spMTX >= -1)
        assert resMTX.matrix.nnz == 5
        resMTX = spMTX.multiply(spMTX < -1)
        assert resMTX.matrix.nnz == 3
        resMTX = spMTX.multiply(spMTX <= -1)
        assert resMTX.matrix.nnz == 4
        resMTX = spMTX.multiply(spMTX > 0)
        assert resMTX.matrix.nnz == 3
        resMTX = spMTX.multiply(spMTX < 0)
        assert resMTX.matrix.nnz == 4

        # test normal

    def test_empty_rows(self, spMTX, nmMTX, sqMTX):
        pass

    def test_equal(self, spMTX, nmMTX, sqMTX):
        # test sparse
        otherMTX = spMTX.copy()
        assert spMTX.equal(otherMTX)
        otherMTX.matrix[1, 3] = -3
        assert not spMTX.equal(otherMTX)

        # test normal
        otherMTX = nmMTX.copy()
        assert nmMTX.equal(otherMTX)
        otherMTX.matrix[1, 3] = -3
        assert not nmMTX.equal(otherMTX)

        # test square
        otherMTX = sqMTX.copy()
        assert sqMTX.equal(otherMTX)
        otherMTX.matrix[1, 3] = -3
        assert not sqMTX.equal(otherMTX)

    def test_most_similar(self, spMTX, nmMTX, sqMTX):
        simeles = sqMTX.most_similar('col1', k=2)
        assert simeles == ['col0', 'col2']

    def test_to_csv(self, spMTX, nmMTX, sqMTX):
        #spMTX.to_csv()
        pass

    def test_read_csv(self, spMTX, nmMTX, sqMTX):
        #TypeTokenMatrix.read_csv()
        pass

    def test_concatenate(self, spMTX, nmMTX, sqMTX):
        spMTX2 = spMTX.copy()
        spMTX2.row_items = ['row3', 'row4', 'row5']
        concspMTX = spMTX.concatenate(spMTX2, axis=0)
        assert concspMTX.row_items == (spMTX.row_items + spMTX2.row_items)
        assert concspMTX.col_items == spMTX.col_items
        assert concspMTX.matrix != sp.hstack([spMTX.matrix, spMTX2.matrix])

        spMTX2 = spMTX.copy()
        spMTX2.col_items = ['col4', 'col5', 'col6', 'col7']
        concspMTX = spMTX.concatenate(spMTX2, axis=1)
        assert concspMTX.row_items == spMTX.row_items
        assert concspMTX.col_items == (spMTX.col_items + spMTX2.col_items)
        assert concspMTX.matrix != sp.vstack([spMTX.matrix, spMTX2.matrix])