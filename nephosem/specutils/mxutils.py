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

import math
import operator
import logging

import numpy as np
import scipy.sparse as sp

from six import iteritems

from nephosem.core import matrix

__all__ = ['transform_dict_to_spmatrix', 'transform_spmatrix_to_dict',
           'transform_nodes_to_matrix', 'transform_indices',
           'merge_two_matrices', 'merge_matrices']

logger = logging.getLogger(__name__)


def transform_dict_to_spmatrix(dict_mtx, rowid2item, colid2item, verbose=False):
    """Generate sparse.csr_matrix from dict of dict,
    according to row item to id and col item to id mappings

    Parameters
    ----------
    dict_mtx : dict of dict
        Matrix represented by a Python dict of dict
    rowid2item : list of str
        Alphabetically ascending sorted list of items
    colid2item : list of str
        Alphabetically ascending sorted list of items
    verbose : bool

    Returns
    -------
    sparse.csr_matrix
    """
    if verbose:
        logger.info("\nBuilding sparse matrix from python dict of dict...")
    item2rowid = {e: i for i, e in enumerate(rowid2item)}  # item -> index mapping
    item2colid = {e: i for i, e in enumerate(colid2item)}
    nRows = len(item2rowid)
    nCols = len(item2colid)
    if verbose:
        logger.info("  Rows: %s, cols: %s" % (nRows, nCols))
    row, col = [], []
    data = []

    # build indices and data
    for rowitem, vec in sorted(dict_mtx.items(), key=operator.itemgetter(0)):
        for colitem, value in sorted(vec.items(), key=operator.itemgetter(0)):
            row.append(item2rowid[rowitem])  # row index
            col.append(item2colid[colitem])  # col index
            data.append(value)
    if len(data) == 0:
        raise ValueError("Error: the input matrix is empty!!!")
    dt = type(data[0])
    ''' TODO: check data types
    dtypes = {
        int: np.int32,
        float: np.float64,
        np.int32: np.int32,
        np.int64: np.int64,
        np.float32: np.float32,
        np.float64: np.float64,
    }
    if dt not in dtypes:
       raise ValueError("Error: unsupported data type!!!{}".format(str(dt)))
    row = np.array(row, dtype=dtypes[dt])
    col = np.array(col, dtype=dtypes[dt])
    data = np.array(data, dtype=dtypes[dt])
    '''
    row = np.array(row, dtype=dt)
    col = np.array(col, dtype=dt)
    data = np.array(data, dtype=dt)

    nEles = len(data)
    if verbose:
        logger.info("  Done... Num of data: {}\n".format(nEles))

    return sp.csr_matrix((data, (row, col)), shape=(nRows, nCols))


def transform_spmatrix_to_dict(spmatrix, rowid2item, colid2item, verbose=False):
    """Generate dict of dict from sparse.csr_matrix,
    according to row item to id and col item to id mappings

    Parameters
    ----------
    spmatrix : :class:`~scipy.sparse.csr_matrix`
    rowid2item : iterable (of str)
        Alphabetically ascending sorted list of items
    colid2item : iterable (of str)
        Alphabetically ascending sorted list of items
    verbose : bool

    Returns
    -------
    Python dict of dict
    """
    if verbose:
        print("\nBuilding python dict of dict from sparse matrix...")

    dict_mtx = dict()
    matrix = spmatrix.tocoo()  # easier usage
    row, col, data = matrix.row, matrix.col, matrix.data
    for i in range(len(data)):
        rid, cid = row[i], col[i]
        val = data[i]
        rowitem, colitem = rowid2item[rid], colid2item[cid]
        if rowitem not in dict_mtx:
            dict_mtx[rowitem] = dict()
        dict_mtx[rowitem][colitem] = val

    return dict_mtx


def transform_nodes_to_matrix(type2toks, colloc_fmt = 'lemma/pos'):
    """Transform type nodes to token matrix.

    Parameters
    ----------
    type2toks : dict or iterable
        Type string -> token nodes of this type
    colloc_fmt : str, default="lemma/pos"
        Format for the column names

    Returns
    -------
    tokmx : :class:`~nephosem.core.matrix.TypeTokenMatrix`
    """
    if not isinstance(type2toks, dict):
        if not isinstance(type2toks, list):
            raise ValueError("Please pass a dict or a list")
        type2toks = {str(tp): tp for tp in type2toks}
    # transform type2toks dict into token_context boolean matrix
    tok2collocs = {}
    col_collocs = set()
    for tpstr, tpnode in type2toks.items():
        for tok in tpnode.tokens:
            cleft = {colloc.to_colloc(colloc_fmt=colloc_fmt): (i - len(tok.lcollocs))
                     for i, colloc in enumerate(tok.lcollocs)}
            cright = {colloc.to_colloc(colloc_fmt=colloc_fmt): i + 1
                      for i, colloc in enumerate(tok.rcollocs)}
            tok2collocs[str(tok)] = {**cleft, **cright}
            col_collocs = col_collocs.union(set(tok2collocs[str(tok)].keys()))
    row_items = list(tok2collocs.keys())
    col_items = list(col_collocs)
    tokmx = transform_dict_to_spmatrix(tok2collocs, row_items, col_items)
    return matrix.TypeTokenMatrix(tokmx, row_items, col_items)


def merge_two_matrices(mtx1, mtx2):
    """Merge two (TypeTokenMatrix) matrices.

    Parameters
    ----------
    mtx1 : :class:`~nephosem.TypeTokenMatrix`
    mtx2 : :class:`~nephosem.TypeTokenMatrix`

    Returns
    -------
    merged matrix : :class:`~nephosem.TypeTokenMatrix`
    """
    if mtx1.__class__ != mtx2.__class__:
        raise ValueError("The given two matrices are not the same type!")

    # check whether these two matrices have same row items and column items
    consistent = [True, True]
    if mtx1.row_items != mtx2.row_items:
        consistent[0] = False
    if mtx1.col_items != mtx2.col_items:
        consistent[1] = False

    if consistent[0] and consistent[1]:
        # these matrices have same row items and column items
        # then just add the scipy sparse matrices
        spmtx = mtx1.matrix + mtx2.matrix
        return mtx1.__class__(spmtx, mtx1.row_items, mtx1.col_items)
    else:
        if not consistent[0]:  # row items inconsistent
            # row_items = sorted(set(mtx1.row_items) + set(mtx2.row_items))
            # rowmapping = {v:k for k, v in enumerate(row_items)}
            raise NotImplementedError
        if not consistent[1]:  # col items inconsistent
            # union the two column item lists
            col_items = sorted(set(mtx1.col_items).union(set(mtx2.col_items)))
            # transform two sparse matrices
            spmx1 = transform_indices(mtx1, col_items)
            spmx2 = transform_indices(mtx2, col_items)
            spmx = spmx1 + spmx2
            return mtx1.__class__(spmx, mtx1.row_items, col_items)


def transform_indices(mtx, new_col_items):
    """"""
    spmx = mtx.matrix
    glbitem2glbidx = {v: k for k, v in enumerate(new_col_items)}
    subidx2glbidx = [glbitem2glbidx[it] for i, it in enumerate(mtx.col_items)]
    # time cost bottleneck (the size of matrix.indices is large)
    newindices = np.array([subidx2glbidx[i] for i in spmx.indices])
    newspmx = sp.csr_matrix((spmx.data, newindices, spmx.indptr), shape=(spmx.shape[0], len(new_col_items)))
    return newspmx


def merge_matrices(matrices):
    """Merge a list of (TypeTokenMatrix) matrices into one.

    Parameters
    ----------
    matrices : list
        A list of matrices (:class:`~nephosem.TypeTokenMatrix`)

    Returns
    -------
    tuple :
        spmatrix, row_items, col_items
    """
    if not matrices or len(matrices) == 0:
        return matrices
    if len(matrices) == 1:
        return matrices[0].matrix, matrices[0].row_items, matrices[0].col_items

    consistent = True
    # check row items and column items consistency
    row_items, col_items = matrices[0].row_items, matrices[0].col_items
    for i in range(1, len(matrices)):
        if matrices[i].row_items != row_items:
            consistent = False
        if matrices[i].col_items != col_items:
            consistent = False

    if consistent:
        # since row items and column items are the same for all sub-matrices
        # we can use scipy sparse matrix method to merge matrices
        spmx = matrices[0].matrix.copy()
        for i in range(1, len(matrices)):
            spmx += matrices[i].matrix
        return spmx, row_items, col_items
    else:  # transform sub-matrices to Python dict of dict and merge
        mxdict_list = []
        row_items, col_items = set(), set()
        for mx in matrices:
            mxdict = transform_spmatrix_to_dict(mx.matrix, mx.row_items, mx.col_items)
            mxdict_list.append(mxdict)
            row_items = row_items | set(mx.row_items)  # construct union row items
            col_items = col_items | set(mx.col_items)  # construct union column items

        mxdict = merge_matrix_dict(mxdict_list)
        row_items = sorted(row_items)  # transform dict to list
        col_items = sorted(col_items)
        spmx = transform_dict_to_spmatrix(mxdict, row_items, col_items)
        return spmx, row_items, col_items


def merge_matrix_dict(mxdict_list):
    """Merge a list of matrices (dict of dict)

    Parameters
    ----------
    mxdict_list : list
        A list of matrices (dict of dict)
    """
    matrix = dict()
    for mxdict in mxdict_list:
        for rk, row in iteritems(mxdict):
            if rk not in matrix:
                matrix[rk] = dict()
            for ck, val in iteritems(row):
                if ck not in matrix[rk]:
                    matrix[rk][ck] = 0
                matrix[rk][ck] += val
    return matrix


def get_largest_k(arr, k):
    """Get indices of the largest k elements.
    The indices are not ordered.
    e.g. indices = array([1, 2]) does not mean that
    the value of index 1 is larger than the value of index 2
    """
    if k >= arr.shape[0]:
        return np.arange(arr.shape[0])
    indices = np.argpartition(arr, -k)[-k:]  # see usage of numpy.argpartition
    return indices


def get_smallest_k(arr, k):
    """Get indices of the smallest k elements.
    The indices are not ordered.
    """
    if k >= arr.shape[0]:
        return np.arange(arr.shape[0])
    indices = np.argpartition(arr, k)[:k]
    return indices


def centroid_of_cluster(data, cluster=None):
    """Compute the centroid of a cluster of the data
    if the cluster is not None,
    then the data should be the whole matrix of all elements,
    and the centroid is computed on the cluster
    otherwise, the centroid is computed on all elements

    Parameters
    ----------
    data : a numpy.ndarray
    cluster : array of id
    """
    mtx = data
    if cluster is not None:
        if not isinstance(cluster, np.ndarray):
            cluster = np.array(cluster)
        mtx = mtx[cluster, :]   # this code creates a new array and may be slow
    cent = np.mean(mtx, axis=0)
    return cent


def sum_of_cluster(data, cluster=None):
    mtx = data
    if cluster is not None:
        if not isinstance(cluster, np.ndarray):
            cluster = np.array(cluster)
        mtx = mtx[cluster, :]
    return np.sum(mtx, axis=0)
