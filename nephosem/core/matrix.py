#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: QLVL <qlvl@kuleuven.be>
# Copyright (C) 2021 QLVL KULeuven
#
# This file is part of Nephosem.
#
# Nephosem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Nephosem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nephosem. If not, see <https://www.gnu.org/licenses/>.


"""Matrix Classes

Usage examples
==============

Construct a TypeTokenMatrix with a Python dict e.g.

>>> from nephosem.tests.utils import common_texts
>>> from nephosem import TypeTokenMatrix
>>>
>>>

"""

import codecs
import json
import logging
import os
import zipfile

import random

try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED
from copy import deepcopy
from tabulate import tabulate

import numpy as np
import pandas as pd
import scipy.sparse as sp

from nephosem.specutils import mxutils
# import transform_spmatrix_to_dict, transform_dict_to_spmatrix, get_largest_k, get_smallest_k, centroid_of_cluster
from nephosem.utils import is_string

__all__ = ['TypeTokenMatrix']

logger = logging.getLogger(__name__)


def check_symmetric(arr, tol=1e-8):
    if arr.shape[0] != arr.shape[1]:
        return False
    return np.allclose(arr, arr.T, atol=tol)


def empty_row_idx(npmx):
    empty_idx = []
    for i in range(npmx.shape[0]):
        row = npmx[i]
        nonzeros = np.count_nonzero(row)
        if nonzeros == 0:
            empty_idx.append(i)
    return np.array(empty_idx)


def merges_matrices(matrix_list):
    raise NotImplementedError


class BaseMatrix(object):
    def get_matrix(self):
        """Get matrix.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix:

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def transpose(self):
        """Computes the transpose of a matrix.

        Returns
        -------
        numpy.ndarray:
            The transpose.

        """
        raise NotImplementedError

    def svd(self, k=7, ascending=False):
        """Computes the first k singular value decomposition vectors.

        Parameters
        ----------
        k : int
            Number of principal components.
        ascending : bool
            If ascending is set to True, compute the last k SVD vectors.

        """
        raise NotImplementedError


def reorder_matrix(matrix, idx, axis=0):
    """Reorder the matrix based on a new item index sequence.

    Parameters
    ----------
    matrix : scipy.spmatrix or numpy.ndarray
    idx : list or numpy.ndarray
        numpy.ndarray is faster.
    axis : int, optional

    """
    if isinstance(idx, np.ndarray):
        idx = np.array(idx)
    if axis == 0:
        matrix = matrix[idx]
    elif axis == 1:
        matrix = matrix[:, idx]
    else:
        raise ValueError("Axis should be 0 or 1!")
    return matrix


class SparseMatrix(BaseMatrix):
    """Sparse Matrix."""
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def dataframe(self):
        # return pd.SparseDataFrame(self.matrix).to_dense()
        # The code is outdated. Use the following instead.
        return pd.DataFrame.sparse.from_spmatrix(self.matrix).sparse.to_dense()


    def __getitem__(self, arg):
        return self._get_value(arg)

    def _get_value(self, arg):
        """Equivalent of calling the matrix directly, e.g. self[arg]

        Parameters
        ----------
        arg : items, integers, lists or slices.
            If tuple, it represents rows and columns respectively.

        Returns
        -------
        If a tuple of strings or integers, it returns the value in its place.
        Otherwise a sliced matrix.

        """
        val = self.matrix[arg]
        if isinstance(val, sp.spmatrix):
            val = val.toarray()
        if len(val.shape) > 1 and val.shape[0] == 1:
            val = val[0]
        return val

    def multiply(self, other):
        """

        Parameters
        ----------
        other : spmatrix

        """
        nonzeros = np.count_nonzero(self.matrix.data == 0)
        if nonzeros > 0:  # sparse matrix has zero values
            coomx = self.matrix.tocoo(copy=True)
            row, col, data = coomx.row, coomx.col, coomx.data
            othercoo = other.tocoo(copy=True)
            mask_pairs = {(r, c) for r, c in zip(othercoo.row, othercoo.col)}
            masked_row, masked_col, masked_data = [], [], []
            for r, c, d in zip(row, col, data):
                if (r, c) in mask_pairs:
                    masked_row.append(r)
                    masked_col.append(c)
                    masked_data.append(d)
            masked_row, masked_col, masked_data = np.array(masked_row), np.array(masked_col), np.array(masked_data)
            masked_mtx = sp.csr_matrix(
                            (masked_data, (masked_row, masked_col)),
                            shape=self.matrix.shape)
            return masked_mtx
        else:
            return self.matrix.multiply(other)

    def __lt__(self, other):
        if other <= 0.0:
            resmx = self.matrix < other
        else:
            resmx = self._operator_positive(other, operator='lt')
        return resmx

    def __le__(self, other):
        if other <= 0.0:
            resmx = self.matrix <= other
        else:
            resmx = self._operator_positive(other, operator='le')
        return resmx

    def __gt__(self, other):
        if other >= 0.0:
            resmx = self.matrix > other
        else:
            resmx = self._operator_positive(other, operator='gt')
        return resmx

    def __ge__(self, other):
        if other >= 0.0:
            resmx = self.matrix >= other
        else:
            resmx = self._operator_positive(other, operator='ge')
        return resmx

    def _operator_positive(self, val, operator='lt'):
        """Operations with a positive value."""
        spmx = self.matrix.tocoo(copy=True)
        row, col, data = spmx.row, spmx.col, spmx.data
        shape = spmx.shape

        if operator == 'lt':
            mask = data < val
        elif operator == 'le':
            mask = data <= val
        elif operator == 'gt':
            mask = data > val
        elif operator == 'ge':
            mask = data >= val
        else:
            raise NotImplementedError("Error: unimplemented operator!")

        row = row[mask]
        col = col[mask]
        data = np.ones(row.shape, dtype=bool)
        spmx = sp.csr_matrix((data, (row, col)), shape=shape)
        return spmx

    def submatrix(self, row=None, col=None):
        """submatrix based on row indices or column indices.

        Parameters
        ----------
        row : list or numpy.ndarray
        col : list or numpy.ndarray

        Returns
        -------
        scipy.sparse.csr_matrix

        """
        # TODO: may have problem with explicit zeros
        if row is not None:
            if col is not None:
                # submatrix on both row and column
                submx = self.matrix[row, :]
                submx = submx[:, col]
                # slicing on columns of a csc_matrix is faster (seems not)
                # submx = submx.tocsc()
                # submx = submx[:, col].tocsr()
            else:
                submx = self.matrix[row, :]  # select rows
        elif col is not None:
            submx = self.matrix[:, col]
            # transform to csc?
            # no need, even slower than directly slicing it
            # submx = self.matrix.tocsc()
            # submx = submx[:, col].tocsr()  # select columns
        else:
            # no submatrix to select
            submx = self.matrix
        return submx

    def reorder(self, idx, axis=0):
        """Reorder the matrix based on a new item index sequence.

        Parameters
        ----------
        idx : list or numpy.ndarray
            numpy.ndarray is faster.
        axis : int, optional

        """
        return reorder_matrix(self.matrix, idx, axis=axis)

    def controid(self, cluster=None):
        """Calculate the centroid value of the cluster
        if the cluster is not provided, calculate the centroid of the whole matrix

        Parameters
        ----------
        cluster : array of element ids

        Returns
        -------
        centroid vector

        """
        data = self.matrix.toarray()
        cent = mxutils.centroid_of_cluster(data, cluster)
        return cent

    def empty_rows(self, explicit=True):
        """Show types/tokens which have empty rows
        If this matrix is a token-context weight matrix,
        the row of a token would have all zero ppmi values.

        Parameters
        ----------
        explicit : bool
            If True, rows that only have explicit zeros are empty.
            Else, rows that have no stored values are empty.

        Returns
        -------
        a list of row indices

        """
        if explicit:
            # get rows that has stored values, including explicit zeros
            row_mask = self.matrix.getnnz(1) > 0  # -> boolean array of row size
            row_mask = ~row_mask  # reverse True <-> False
            row_idx = np.arange(self.matrix.shape[0])
            empty_idx = row_idx[row_mask]
        else:
            # common for numpy.ndarray
            empty_idx = empty_row_idx(self.matrix.toarray())
        return empty_idx

    def drop_zero_rows(self):
        """Drop rows with only zero values and return the dropped matrix"""
        if not isinstance(self.matrix, sp.coo_matrix):
            coomx = self.matrix.to_coo()
        else:
            coomx = self.matrix
        nz_rows, new_row = np.unique(coomx.row, return_inverse=True)
        new_csr = sp.csr_matrix((coomx.data, (new_row, coomx.col)), shape=(len(nz_rows), coomx.shape[1]))
        return new_csr

    def drop_zero_cols(self):
        """Drop columns with only zero values and return the dropped matrix"""
        if not isinstance(self.matrix, sp.coo_matrix):
            coomx = self.matrix.to_coo()
        else:
            coomx = self.matrix
        nz_cols, new_col = np.unique(coomx.col, return_inverse=True)
        new_csr = sp.csr_matrix((coomx.data, (coomx.row, new_col)), shape=(coomx.shape[0], len(nz_cols)))
        return new_csr

    def most_similar(self, rowid, k=10, metric='cosine', descending=False):
        raise NotImplementedError

    def merge(self, self_row_items, self_col_items, othermx, other_row_items, other_col_items):
        """Merge two sparse matrices.

        Parameters
        ----------
        self_row_items
        self_col_items
        othermx
        other_row_items
        other_col_items

        Returns
        -------

        """
        mxdict1 = mxutils.transform_spmatrix_to_dict(self.matrix, self_row_items, self_col_items)
        mxdict2 = mxutils.transform_spmatrix_to_dict(othermx, other_row_items, other_col_items)
        row_items = sorted(set(self_row_items) | set(other_row_items))
        col_items = sorted(set(self_col_items) | set(other_col_items))
        mxdict = self._merge_matrix_dict(mxdict1, mxdict2)
        spmx = mxutils.transform_dict_to_spmatrix(mxdict, row_items, col_items)
        return spmx, row_items, col_items

    @classmethod
    def _merge_matrix_dict(cls, matrix1, matrix2):
        """Merge two matrix (dict of dict)

        Parameters
        ----------
        matrix1 : dict of dict
        matrix2 : dict of dict

        """
        matrix = dict()
        matrix.update(matrix1)
        for rk, row in matrix2.items():
            if rk not in matrix:
                matrix[rk] = dict()
            for ck, val in row.items():
                if ck not in matrix[rk]:
                    matrix[rk][ck] = 0
                matrix[rk][ck] += val
        return matrix

    def concatenate(self, othermx, axis=0):
        """Concatenate the other matrix with self.

        Parameters
        ----------
        othermx : scipy.sparse.csr_matrix
        axis : int
        """
        if axis == 0:
            return sp.vstack([self.matrix, othermx])
        elif axis == 1:
            return sp.hstack([self.matrix, othermx])
        else:
            raise ValueError("axis should be 0 or 1!")

    def transpose(self):
        return self.matrix.transpose()

    def equal(self, othermx):
        if self.matrix.shape != othermx.shape or self.matrix.nnz != othermx.nnz:
            return False
        return (self.matrix != othermx).nnz == 0

    def get_colloc_contexts(self, idx):
        if not isinstance(self.matrix, sp.csr_matrix):
            spmx = self.matrix.tocsr()
        else:
            spmx = self.matrix
        indptr, indices, data = spmx.indptr, spmx.indices, spmx.data
        col_idx = indices[indptr[idx]:indptr[idx + 1]]
        return col_idx


class NormalMatrix(BaseMatrix):
    """Normal Matrix."""
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def dataframe(self):
        return pd.DataFrame(self.matrix)

    def __getitem__(self, arg):
        return self._get_value(arg)

    def _get_value(self, arg):
        return self.matrix[arg]

    def multiply(self, other):
        return self.matrix * other

    def __lt__(self, other):
        return self.matrix < other

    def __le__(self, other):
        return self.matrix <= other

    def __gt__(self, other):
        return self.matrix > other

    def __ge__(self, other):
        return self.matrix >= other

    def reorder(self, idx, axis=0):
        """Reorder the matrix based on a new item index sequence.

        Parameters
        ----------
        idx : list or numpy.ndarray
            numpy.ndarray is faster.
        axis : int, optional

        """
        return reorder_matrix(self.matrix, idx, axis=axis)

    def submatrix(self, row=None, col=None):
        """

        Parameters
        ----------
        row : list of int or numpy.ndarray
        col : list of int or numpy.ndarray

        Returns
        -------

        """
        if row is not None:
            if col is not None:
                submx = self.matrix[row]
                submx = submx[:, col]
            else:
                submx = self.matrix[row]
        elif col is not None:
            submx = self.matrix[:, col]
        else:
            raise ValueError
        return submx

    def empty_rows(self, explicit=True):
        raise NotImplementedError

    def merge(self, *args):
        raise NotImplementedError

    def concatenate(self, othermx, axis=0):
        """Concatenate the other matrix with self.

        Parameters
        ----------
        othermx : scipy.sparse.csr_matrix
        axis : int
        """
        if axis == 0 or axis == 1:
            return np.concatenate((self.matrix, othermx), axis=axis)
        else:
            raise ValueError("axis should be 0 or 1!")

    def transpose(self):
        return self.matrix.transpose()

    def equal(self, othermx):
        if self.matrix.shape != othermx.shape:
            return False
        return np.allclose(self.matrix, othermx)

    def get_colloc_contexts(self, idx):
        raise NotImplementedError

    def most_similar(self, rid, k=10, descending=False):
        raise NotImplementedError


class SquareMatrix(BaseMatrix):
    """Square Matrix."""
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def dataframe(self):
        return pd.DataFrame(self.matrix)

    def __getitem__(self, arg):
        return self._get_value(arg)

    def _get_value(self, arg):
        return self.matrix[arg]

    def multiply(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def reorder(self, idx, axis=0):
        raise NotImplementedError("SquareMatrix should not be reordered.")

    def submatrix(self, row=None, col=None):
        """Select a square matrix based on the input idx (indices).
        The rows in the idx and columns in the idx will be selected.

        Parameters
        ----------
        idx : list of int or numpy.ndarray

        Returns
        -------

        """
        submx = self.matrix[row]
        submx = submx[:, col]
        return submx

    def merge(self, *args):
        raise NotImplementedError

    def concatenate(self, *args, **kwargs):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError("Symmetric matrix doesn't need to be transposed!")

    def equal(self, othermx):
        if self.matrix.shape != othermx.shape:
            return False
        return np.allclose(self.matrix, othermx)

    def get_colloc_contexts(self, idx):
        raise NotImplementedError

    def most_similar(self, rid, k=10, descending=False):
        """Get most similar items of the target item.

        Parameters
        ----------
        rid : int
            Row index (of item)
        k : int
            Number of returned similar items.
        descending : bool
            If descending is True, sort the elements according to descending order of the values.
            Else, sort the elements according to ascending order of the values.
            The values would be distance or similarity.
            So for similarity matrix, set `descending` to True, as we want elements with largest values (similarities).
            For distance matrix, set `descending` to False.
            For similarity rank matrix, same to similarity matrix.

        Returns
        -------
        a list of elements

        """
        row = self.matrix[rid]
        if descending:  # for similarity matrix
            idx = mxutils.get_largest_k(row, k+1)
        else:
            idx = mxutils.get_smallest_k(row, k+1)
        return idx


class TypeTokenMatrix(BaseMatrix):
    """
    Examples
    --------
    Construction of a toy TypeTokenMatrix object:

    >>> row_items = ['row0', 'row1', 'row2']
    >>> col_items = ['col0', 'col1', 'col2', 'col3']

    >>> sparr = np.array([[-5, 0, -3, -2], [-1, 0, 0, 1], [2, 0, 4, 5]])
    >>> spMTX = TypeTokenMatrix(spmx, row_items, col_items)
    >>> print(spMTX)

    >>> dsmx = np.array([[-5, -4, -3, -2], [-1, 0, 0, 1], [2, 3, 4, 5]])
    >>> nmMTX = TypeTokenMatrix(dsmx, row_items, col_items)
    >>> print(nmMTX)

    >>> sqarr = np.array([1, 2, 3, 4, 5, 6])
    >>> from scipy.spatial.distance import squareform
    >>> sqmx = squareform(sqarr)
    >>> sqMTX = TypeTokenMatrix(sqmx, col_items, col_items)
    >>> print(sqMTX)

    """
    def __init__(self, matrix, row_items, col_items, deep=True, **kwargs):
        if deep:
            self.matrix = deepcopy(matrix)
        else:  # if the passed matrix is already a new object before passing in
            self.matrix = matrix
        # check validity of matrix with row_items and col_items
        if matrix.shape[0] != len(row_items):
            raise ValueError("Inconsistent size of row item list ({}) with shape of matrix ({})!"
                             .format(len(row_items), self.matrix.shape[0]))
        if matrix.shape[1] != len(col_items):
            raise ValueError("Inconsistent size of column item list ({}) with shape of matrix ({})!"
                             .format(len(col_items), self.matrix.shape[1]))
        self._mxbehavior = None
        # always deep copy new row&column items, to keep them save
        self.row_items = deepcopy(row_items)
        self.col_items = deepcopy(col_items)
        self._item2rowid = None
        self._item2colid = None

        if isinstance(self.matrix, sp.spmatrix):
            self._mxbehavior = SparseMatrix(self.matrix)
        elif isinstance(self.matrix, np.ndarray):
            if check_symmetric(self.matrix):
                if self.row_items != self.col_items:
                    raise ValueError("The row items and column items are not equal!")
                self._mxbehavior = SquareMatrix(self.matrix)
            else:
                self._mxbehavior = NormalMatrix(self.matrix)
        else:
            raise NotImplementedError("Not support this type of matrix!")

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def rowid2item(self):
        return self.row_items

    @property
    def colid2item(self):
        return self.col_items

    @property
    def item2rowid(self):
        """Return a dict mapping from (row) items to corresponding indices.
        If `self._item2rowid` is generated, just return it,
        else, create the `self._item2rowid` dict and return it.

        Returns
        -------
        self._item2rowid : dict
        """
        if self._item2rowid is None:
            self._item2rowid = {e: i for i, e in enumerate(self.rowid2item)}
        return self._item2rowid

    @property
    def item2colid(self):
        """Return a dict mapping from (column) items to corresponding indices.
        If `self._item2colid` is generated, just return it,
        else, create the `self._item2colid` dict and return it.

        Returns
        -------
        self._item2colid : dict
        """
        if self._item2colid is None:
            self._item2colid = {e: i for i, e in enumerate(self.colid2item)}
        return self._item2colid

    @property
    def meta_data(self):
        """Meta data of the matrix."""
        meta = dict()
        notmeta = ['matrix', 'row_items', 'col_items', '_mxbehavior']
        for k, v in self.__dict__.items():
            if k not in notmeta:
                meta[k] = v
        return meta

    @property
    def dataframe(self):
        df = self._mxbehavior.dataframe
        df.rename(index={i: e for i, e in enumerate(self.row_items)},
                  columns={i: e for i, e in enumerate(self.col_items)},
                  inplace=True)
        return df

    def get_matrix(self):
        return self.matrix.copy()

    def __getitem__(self, arg):
        return self._get_value(arg)

    def _get_value(self, arg):
        item2rowid = self.item2rowid
        item2colid = self.item2colid
        # check arg of these cases
        if isinstance(arg, tuple):
            # mtx[1, 2], mtx[1, :10], mtx[:, 2], mtx[:10, :10]
            if len(arg) == 2:
                arg1, arg2 = arg
                if is_string(arg1):
                    # mtx['str1', 2], mtx['str1', :10]
                    arg1 = item2rowid[arg1]
                elif isinstance(arg1, int) or isinstance(arg1, slice):
                    pass
                else:
                    raise ValueError("Insupportable arguments!")
                if is_string(arg2):
                    # mtx['str1', 'str2'], mtx[1, 'str2'], mtx[:10, 'str2']
                    arg2 = item2colid[arg2]
                elif isinstance(arg2, int) or isinstance(arg2, slice):
                    pass
                else:
                    raise ValueError("Insupportable arguments!")
                arg = (arg1, arg2)
            else:
                raise ValueError("If passing a tuple, it should be a 2-tuple.")
        elif isinstance(arg, slice) or isinstance(arg, int):
            # cases: mtx[1], mtx[:10]
            pass
        elif is_string(arg):
            # case: mtx['str']
            arg = item2rowid[arg]
        elif isinstance(arg, list):
            raise NotImplementedError
        else:
            raise ValueError("If only pass a value, it should be an integer!")

        return self._mxbehavior[arg]

    def multiply(self, other):
        # TODO: check consistency first!!!
        resmx = self._mxbehavior.multiply(other.matrix)
        return TypeTokenMatrix(resmx, self.row_items, self.col_items, **self.meta_data)

    def __lt__(self, other):
        if type(other) not in [int, float]:
            raise ValueError("Error: please pass an int or float value!")
        resmx = self._mxbehavior.__lt__(other)
        return TypeTokenMatrix(resmx, self.row_items, self.col_items, **self.meta_data)

    def __le__(self, other):
        if type(other) not in [int, float]:
            raise ValueError("Error: please pass an int or float value!")
        resmx = self._mxbehavior.__le__(other)
        return TypeTokenMatrix(resmx, self.row_items, self.col_items, **self.meta_data)

    def __gt__(self, other):
        if type(other) not in [int, float]:
            raise ValueError("Error: please pass an int or float value!")
        resmx = self._mxbehavior.__gt__(other)
        return TypeTokenMatrix(resmx, self.row_items, self.col_items, **self.meta_data)

    def __ge__(self, other):
        if type(other) not in [int, float]:
            raise ValueError("Error: please pass an int or float value!")
        resmx = self._mxbehavior.__ge__(other)
        return TypeTokenMatrix(resmx, self.row_items, self.col_items, **self.meta_data)

    #def __deepcopy__(self, memodict={}):
    #    pass

    def sum(self, axis=None):
        """Sum the matrix over the given axis.
        If the axis is None, sum over both rows and columns, returning a scalar.

                    axis = 1
        -----------    ^
        | 1, 0, 2 |   [3]
        | 0, 3, 4 |   [7]
        -----------
         [1, 3, 6]        > axis = 0

                      Collocate present   Collocate absent    Totals
        Node present  c_a_b               c_a_nb              R1
        Node absent   c_na_b              c_na_nb             R2
        Totals        C1                  C2                  N

        Parameters
        ----------
        axis : int
            If axis == 1, sum over rows.
            If axis == 0, sum over columns.
            If axis is None, return the total sum value (scalar).

        Returns
        -------
        dict
            A python dict with row/column items as keys and sum of that row/column as values.
        """
        if axis is None:  # when axis = 0, not axis is True
            return self.matrix.sum()
        items = self.row_items if axis == 1 else self.col_items
        # sum of each rows / columns of matrix (normally collocate frequency matrix)
        sums = np.squeeze(np.asarray(self.matrix.sum(axis=axis))).tolist()
        freq_dict = {k: v for k, v in zip(items, sums)}
        return freq_dict

    def submatrix(self, row=None, col=None):
        """Select a submatrix.
        If self is a sparse matrix (i.e. word-context frequency matrix),
        you can either specify only row or only col or both.
        If self is a square matrix (i.e. word-word distance matrix),
        normally you should select a square submatrix, therefore specify both row and col with the same list.

        Parameters
        ----------
        row : iterable (list of str)
            Only support a list of str
        col : iterable (list of str)
            Only support a list of str

        Returns
        -------
        submatrix : :class:`~nephosem.TypeTokenMatrix`
        """
        '''
        row_item_based = False  # -> the input row is a list of item (string)
        col_item_based = False
        if isinstance(row, list):
            if len(row) > 0:
                if is_string(row[0]):
                    row_item_based = True
            else:
                raise ValueError("Please do not pass empty list!")
        if isinstance(col, list):
            if len(col) > 0:
                if is_string(col[0]):
                    col_item_based = True
            else:
                raise ValueError("Please do not pass empty list!")
        if row_item_based:
            item2rowid = self.item2rowid
            row = np.array([item2rowid[e] for e in row])
        if col_item_based:
            item2colid = self.item2colid
            col = np.array([item2colid[e] for e in col])
        '''
        # mapping item strings to indices
        if row is not None:
            item2rowid = self.item2rowid
            requested_rows = deepcopy(row)
            sub_row_items = [e for e in requested_rows if e in item2rowid]
            row = np.array([item2rowid[e] for e in sub_row_items])
            #row = np.array([item2rowid.get(e, 0) for e in row])
            if len(row) == 0:
                logger.warning("The row names provided do not exist.")
                return
            elif len(row) < len(requested_rows):
                lost_rows = len(requested_rows) - len(row)
                logger.warning("{} rows have not been found.".format(str(lost_rows)))
        else:
            sub_row_items = deepcopy(self.row_items)
        if col is not None:
            item2colid = self.item2colid
            requested_cols = deepcopy(col)
            sub_col_items = [e for e in requested_cols if e in item2colid]
            col = np.array([item2colid[e] for e in sub_col_items])
            #col = np.array([item2colid.get(e, 0) for e in col])
            if len(col) == 0:
                logger.warning("The column names provided do not exist.")
                return
            elif len(col) < len(requested_cols):
                lost_cols = len(requested_cols) - len(col)
                logger.warning("{} columns have not been found.".format(str(lost_cols)))
        else:
            sub_col_items = deepcopy(self.col_items)
        submx = self._mxbehavior.submatrix(row=row, col=col)
        # since we generate a new sub-matrix object, we need not deep copy when generating a new TypeTokenMatrix object
        return TypeTokenMatrix(submx, sub_row_items, sub_col_items, deep=False, **self.meta_data)

    def most_similar(self, item, k=10, descending=False):
        """Get most similar items of the target item.

        Parameters
        ----------
        item : str
            Row item (word)
        k : int
            Number of returned similar items.
        descending : bool
            If descending is True, sort the elements according to descending order of the values.
            Else, sort the elements according to ascending order of the values.
            The values would be distance or similarity.
            So for similarity matrix, set `descending` to True, as we want elements with largest values (similarities).
            For distance matrix, set `descending` to False.
            For similarity rank matrix, same to similarity matrix.

        Returns
        -------
        a list of elements
        """
        rid = self.row_items.index(item)  # item should be in row items!
        idx = self._mxbehavior.most_similar(rid, k=k, descending=descending)
        # transform idx back to items
        similars = [self.row_items[i] for i in idx if self.row_items[i] != item]
        return similars

    def sample(self, percent=0.1, seed=-1, replace=False):
        """Sample the matrix based on row

        Parameters
        ----------
        percent : float
            percentage of row dimension
        seed : int, default
            Random seed form sampling.
            When the seed is set to a non-default value (non-negative),
            the method uses this seed for numpy random sampling operation.
        """
        # sample (row) idx
        size = len(self.row_items)
        if percent >= 1:
            if percent > size:
                raise ValueError("'percent' larger than row dimension!")
            sample_size = percent
        elif percent > 0:
            sample_size = int(size * percent)
            sample_size = sample_size if sample_size > 0 else 1
        else:
            raise ValueError("'percent' should be a float between (0, 1) or an integer!")
        if seed >= 0:
            np.random.seed(seed)
        sample_idx = sorted(np.random.choice(range(size), size=sample_size, replace=replace))
        sample_items = [self.row_items[i] for i in sample_idx]
        return self.submatrix(row=sample_items)

    def reorder(self, item_list, axis=0):
        """Reorder the matrix based on a new item list.

        Parameters
        ----------
        item_list : list of str
            A list of (string) items.
        axis : int, optional
            0 for row, 1 for column

        Returns
        -------

        """
        if axis == 0:
            items = self.row_items
        elif axis == 1:
            items = self.col_items
        else:
            raise ValueError("Axis should be 0 or 1!")

        if item_list is not None:
            # item -> id : current item to index mapping
            item2id = {e: i for i, e in enumerate(items)}
            idx_list = np.array([item2id[e] for e in item_list])
        else:
            raise ValueError("Please provide a valid item list!")

        matrix = self._mxbehavior.reorder(idx_list, axis=axis)
        # matrix = self._reorder_matrix(matrix, idx_list, axis=axis)
        if axis == 0:
            row_items = item_list
            col_items = self.col_items
        else:
            row_items = self.row_items
            col_items = item_list
        return TypeTokenMatrix(matrix, row_items, col_items, **self.meta_data)

    def _reorder_verbose(self, item_list=None, idx_list=None, axis=0):
        """Reorder the matrix based on a new item list.

        Parameters
        ----------
        item_list : list of str, optional
            A list of (string) items.
        idx_list : list of int or numpy.ndarray, optional
            A list of index or a numpy ndarray.
            NOTE: currently not support.
        axis : int
            0 for row, 1 for column

        Returns
        -------

        """
        if axis == 0:
            items = self.row_items
        elif axis == 1:
            items = self.col_items
        else:
            raise ValueError("Axis should be 0 or 1!")

        if item_list is not None:
            if idx_list is not None:
                raise ValueError("Please specify which one to use!")
            else:
                # item -> id : current item to index mapping
                item2id = {e: i for i, e in enumerate(items)}
                idx_list = np.array([item2id[e] for e in item_list])
        elif idx_list is not None:
            raise NotImplementedError("'idx_list' parameter is not suggested now!")
            # item_list = np.array([items[i] for i in idx_list])
        else:
            raise ValueError("Please provide item list!")

        matrix = deepcopy(self.matrix)
        matrix = self._reorder_matrix(matrix, idx_list, axis=axis)
        if axis == 0:
            row_items = item_list
            col_items = self.col_items
        else:
            row_items = self.row_items
            col_items = item_list
        return TypeTokenMatrix(matrix, row_items, col_items, **self.meta_data)

    def empty_rows(self, explicit=False):
        """Show types/tokens which have empty rows
        If this matrix is a token-context weight matrix,
        the row of a token would have all zero ppmi values.

        Parameters
        ----------
        explicit : bool
            If True, rows that only have explicit zeros are also empty.
            Else, rows that have no stored values are empty.

        Returns
        -------
        a list of row indices

        """
        row_idx = self._mxbehavior.empty_rows(explicit=explicit)
        empty_items = [self.row_items[i] for i in row_idx]
        return empty_items

    def drop_empty(self, axis=0, explicit=False):
        """Drop empty rows and return a dropped matrix.

        Parameters
        ----------
        axis : int
            0 or 1
        explicit : bool
            If True, rows that only have explicit zeros are also empty.
            Else, rows that only have no stored values are empty.
        """
        if axis == 0:
            return self.drop_empty_rows(explicit=explicit)
        elif axis == 1:
            tmx = self.transpose()
            dropped_tmx = tmx.drop_empty_rows(explicit=explicit)
            return dropped_tmx.transpose()
        else:
            raise ValueError("axis should be 0 or 1!")

    def count_nonzero(self, axis=0):
        """Count the number of nonzero values for each row or each column."""
        nonzeros = (self.matrix != 0).sum(axis=(1-axis))
        nonzeros = np.squeeze(np.asarray(nonzeros))
        return nonzeros

    def drop(self, axis=0, n_nonzero=0, **kwargs):
        """Drop rows which has fewer or equal nonzero values than `n_nonzero`.

        Parameters
        ----------
        axis : int
            If axis is 0, drop rows that satisfy the given criteria.
            If axis is 1, drop columns that satisfy the given criteria.
        n_nonzero : int
            The number of nonzero values in each row.
            If n_nonzero is 0, drop all empty rows.
            If n_nonzero is 1, drop all rows that only have 1 nonzero value or less.
            ...

        Returns
        -------
        Dropped matrix : :class:`~nephosem.TypeTokenMatrix`
        """
        if axis not in [0, 1]:
            raise ValueError("axis should be 0 or 1!")

        if isinstance(self.matrix, np.ndarray):
            raise NotImplementedError("Not support such method for numpy array matrix!")

        nonzeros = self.count_nonzero(axis=axis)
        nonzeros = nonzeros > n_nonzero
        items = self.row_items if axis == 0 else self.col_items
        reset_items = [it for i, it in enumerate(items) if nonzeros[i]]
        if axis == 0:
            return self.submatrix(row=reset_items)
        else:
            return self.submatrix(col=reset_items)

    def drop_empty_rows(self, explicit=False):
        """Drop empty rows and return a dropped matrix.

        Parameters
        ----------
        explicit : bool
            If True, rows that only have explicit zeros are also empty.
            Else, rows that only have no stored values are empty.
        """
        empty_items = self.empty_rows(explicit=explicit)
        if len(empty_items) == 0:
            logger.info("No empty rows!")
            return self
        empty_items = set(empty_items)
        rest_items = [e for e in self.row_items if e not in empty_items]
        return self.submatrix(row=rest_items)

    def drop_zero_rows(self):
        """Drop rows with only zero values and return the dropped matrix"""
        pass

    def merge(self, targetmx):
        """Merge two TypeTokenMatrix objects.

        Parameters
        ----------
        targetmx : :class:`~nephosem.TypeTokenMatrix`
            Target matrix
        """
        merged_mx, merged_row_items, merged_col_items = self._mxbehavior.merge(
            self.row_items, self.col_items,
            targetmx.matrix, targetmx.row_items, targetmx.col_items)
        return TypeTokenMatrix(matrix=merged_mx, row_items=merged_row_items, col_items=merged_col_items)

    def concatenate(self, targetmx, axis=0):
        """Concatenate target matrix with self.

        Parameters
        ----------
        targetmx : :class:`~nephosem.TypeTokenMatrix`
            Target matrix
        axis : int
            Axis of concatenation.
            If axis = 0, concatenate the targetmx as the new rows of self matrix.
            If axis = 1, concatenate the targetmx as the new columns of self matrix.
        """
        if axis == 0:
            concmx = self._mxbehavior.concatenate(targetmx.matrix, axis=axis)
            assert self.col_items == targetmx.col_items
            conc_row_items = self.row_items + targetmx.row_items
            return TypeTokenMatrix(concmx, conc_row_items, self.col_items)
        elif axis == 1:
            concmx = self._mxbehavior.concatenate(targetmx.matrix, axis=axis)
            assert self.row_items == targetmx.row_items
            conc_col_items = self.col_items + targetmx.col_items
            return TypeTokenMatrix(concmx, self.row_items, conc_col_items)
        else:
            raise ValueError("axis should be 0 or 1!")
        pass

    def transpose(self):
        resmx = self._mxbehavior.transpose()
        return TypeTokenMatrix(resmx, self.col_items, self.row_items, **self.meta_data)

    def equal(self, othermx):
        if self.row_items != othermx.row_items:
            return False
        if self.col_items != othermx.col_items:
            return False
        return self._mxbehavior.equal(othermx.matrix)

    def get_colloc_contexts(self, item):
        """Get collocate context features.

        Parameters
        ----------
        item

        Returns
        -------

        """
        if item not in self.row_items:
            raise KeyError("Item {} is not valid!".format(item))
        idx = self.row_items.index(item)
        col_idx = self._mxbehavior.get_colloc_contexts(idx)
        contexts = [self.col_items[i] for i in col_idx]
        return contexts

    def todense(self):
        return TypeTokenMatrix(self.matrix.toarray(), self.row_items, self.col_items, **self.meta_data)

    def copy(self):
        return deepcopy(self)

    def deepcopy(self):
        return deepcopy(self)

    def to_csv(self, filename, sep='\t', index=True, header=True, encoding='utf-8', verbose=True):
        """Write DataFrame to a comma-separated values (csv) file.

        Parameters
        ----------
        filename : str
        sep : character, default '\t'
            Field delimiter for the output file.
        index : boolean, default True
            Write row names (index).
        header : boolean, default True
            Write out the column names.
        encoding : string, optional
            A string representing the encoding to use in the output file, defaults to ‘ascii’ on Python 2 and ‘utf-8’ on Python 3.
        verbose : boolean

        """
        # TODO: float_format : string, default None
        # Format string for floating point numbers
        self.dataframe.to_csv(filename, index=index, header=header, sep=sep, encoding=encoding)

    @classmethod
    def read_csv(cls, filename, sep='\t', index_col=None, header='infer', issparse=False, encoding='utf-8'):
        """Read a comma(tab)-separated values (csv/tsv) file.

        Parameters
        ----------
        filename : str
            Filename of the csv file.
        sep : str, default '\t'
            Field delimiter to use.
        header : int or list of ints, default 'infer'
            Row number(s) to use as the column names, and the start of the data.
        index_col : int or sequence or False, default None
            Column to use as the row labels of the DataFrame.
        issparse : bool
            True for sparse matrix (i.e. frequency matrix).
            False for dense matrix (i.e. distance matrix).
        encoding : str, default 'utf-8'
            Encoding to use for UTF when reading/writing (ex. ‘utf-8’).
        """
        df = pd.read_csv(filename, sep=sep, index_col=index_col, header=header, encoding=encoding)
        return cls.from_dataframe(df, issparse=issparse)

    @classmethod
    def from_dataframe(cls, df, issparse=True):
        row_items = df.index.tolist()
        col_items = df.columns.tolist()
        mx = df.as_matrix()
        if issparse:
            mx = sp.csr_matrix(mx)
        return cls(mx, row_items, col_items)

    def save(self, filename, encoding='utf-8', pack=True, verbose=True):
        meta_data = dict()
        meta_data['row_items'] = self.row_items
        meta_data['col_items'] = self.col_items
        meta_data.update(self.meta_data)

        basename, ext = os.path.splitext(filename)
        if pack:
            if ext != '.pac':
                # filename: '/xxx/BrownNouns.wcmx.freq'
                # not provide '.pac', then assume filename is a base name
                basename = filename
                filename = '{}.pac'.format(basename)
        else:
            # assume filename is a base name
            # filename: '/xxx/BrownNouns.wcmx.freq'
            basename = filename

        meta_fname = "{}.meta".format(basename)
        if verbose:
            logger.info("\nSaving matrix...")

        # save meta data
        with codecs.open(meta_fname, 'w', encoding) as outf:
            json.dump(meta_data, outf, ensure_ascii=False, indent=4)

        if isinstance(self.matrix, sp.spmatrix):
            mx_fname = "{}.npz".format(basename)
            np.savez(mx_fname,
                     data=self.matrix.data,
                     indices=self.matrix.indices,
                     indptr=self.matrix.indptr,
                     shape=self.matrix.shape)
        elif isinstance(self.matrix, np.ndarray):
            mx_fname = "{}.npy".format(basename)
            np.save(mx_fname, self.matrix)
        else:
            raise ValueError

        # package these two files into one named as filename
        if pack:
            with zipfile.ZipFile(filename, 'w', allowZip64=True) as inf:
                inf.write(meta_fname, os.path.basename(meta_fname))
                inf.write(mx_fname, os.path.basename(mx_fname))
            os.remove(meta_fname)
            os.remove(mx_fname)
            if verbose:
                print("Stored in file:\n  {}".format(filename))
        else:
            if verbose:
                print("Stored in files:\n  {}\n  {}".format(meta_fname, mx_fname))

    @classmethod
    def load(cls, filename, encoding='utf-8', pack=True):
        """
        Parameters
        ----------
        filename : ".../xx.wcmx.freq.pac"
        encoding : str
            Default 'utf-8'.
        pack : True or False
            Indicate the file is packaged or not

        Returns
        -------
        meta data and matrix

        """
        basename, ext = os.path.splitext(filename)
        if ext != '.pac':
            basename = filename
        if pack:
            zip_fname = "{}.pac".format(basename)
            dest_dir = os.path.dirname(zip_fname)
            with zipfile.ZipFile(zip_fname) as zf:
                zf.extractall(dest_dir)

        meta_fname = "{}.meta".format(basename)
        sp_fname = "{}.npz".format(basename)
        np_fname = "{}.npy".format(basename)
        issparse = True
        if os.path.exists(sp_fname):
            mx_fname = sp_fname
        elif os.path.exists(np_fname):
            mx_fname = np_fname
            issparse = False
        else:
            raise ValueError("No such matrix file!")

        # TODO: check passed encoding with the encoding in meta data
        # load meta data
        with codecs.open(meta_fname, 'r', encoding) as inf:
            meta_data = json.load(inf)
        if meta_data is None:
            raise(AttributeError("Matrix meta data load error!"))

        # load matrix
        matrix = np.load(mx_fname)
        if issparse:
            matrix = sp.csr_matrix((matrix['data'], matrix['indices'], matrix['indptr']), shape=matrix['shape'])

        # clean files
        if pack:
            os.remove(meta_fname)
            os.remove(mx_fname)

        return TypeTokenMatrix(matrix, meta_data['row_items'], meta_data['col_items'], deep=False)

    def describe(self):
        """Generates descriptive information of the matrix
        TODO: improve
        """
        mtx = self.matrix
        s = list()
        s.append('*******Matrix Description*******')
        s.append('  matrix shape:\t{} X {}'.format(mtx.shape[0], mtx.shape[1]))
        s.append('  is sparse:\t{}'.format(isinstance(self.matrix, sp.spmatrix)))
        s.append('  num of eles:\t{}'.format(mtx.size))
        s.append('  density:\t{0:.2f}%'.format(float(mtx.size * 100) / (mtx.shape[0] * mtx.shape[1])))
        s.append('********************************')

        return '\n'.join(s)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.print_matrix(n_rows=7, n_cols=7)

    def print_matrix(self, n_rows=7, n_cols=7):
        """Prints n_rows and n_cols of the matrix.
        If either is set to None, print the standard amount.

        Parameters
        ----------
        n_rows
        n_cols

        """
        # TODO: improve, split into SparseMatrix, NormalMatrix, SquareMatrix
        # We need to get it to a list of list representation for tabulate
        mtx = self.matrix
        shape = self.shape

        n_rows = min(n_rows, shape[0]) if n_rows else shape[0]
        n_cols = min(n_cols, shape[1]) if n_cols else shape[1]

        # add column words
        res = [[str(list(shape))] + self.colid2item[:n_cols]]

        if isinstance(mtx, sp.spmatrix):
            if not isinstance(mtx, sp.csr_matrix):
                mtx = mtx.tocsr()
            mtx = mtx[:n_rows, :].tocsc()[:, :n_cols].tocoo()
            # add row words
            content = [
                [self.rowid2item[i]] +
                ['NaN'] * n_cols
                for i in range(n_rows)
            ]
            for i, j, val in zip(mtx.row, mtx.col, mtx.data):
                val = '{:0.4f}'.format(val) if isinstance(val, float) else str(val)
                content[i][j+1] = val
            res += content
        elif isinstance(mtx, np.ndarray):
            mtx = mtx[:n_rows, :n_cols]
            res += [
                [self.rowid2item[i]] +
                ['{0:0.4f}'.format(val)
                 if isinstance(val, float)
                 else '{0}'.format(val) for val in row]
                for (i, row) in enumerate(mtx.tolist())
            ]
        else:
            raise ValueError("The matrix should be either scipy.sparse.spmatrix or numpy.ndarray!")

        # add ellipses
        if n_cols < shape[1]:
            [row.append("...") for row in res]
        if n_rows < shape[0]:
            res.append(["..."] * len(res[0]))

        table = tabulate(res, tablefmt='plain')
        return table

    def spmatrix_to_dict(self):
        """Only for sparse matrix"""
        return transform_spmatrix_to_dict(self.matrix, self.row_items, self.col_items)
