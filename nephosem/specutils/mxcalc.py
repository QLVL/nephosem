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

"""Calculations of Matrices"""

import logging
import math
import operator

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing # addition by Stefano (2021.02.18)

from nephosem import progbar
from nephosem.utils import timeit

__all__ = ['compute_association', 'compute_ppmi', 'compute_distance', 'compute_cosine', 'compute_simrank',
           'compute_token_weights', 'compute_token_vectors']

logger = logging.getLogger(__name__)


@timeit
def compute_ppmi(freqMTX, nfreq=None, cfreq=None, positive=True):
    """This method is faster than `compute_association()`.
    Set positive to False to get pmi values.
    """
    tot_sum = float(nfreq.sum())
    freqmx = freqMTX.matrix
    # select frequencies of row items / column items of freqMTX
    # from nfreq and cfreq
    if nfreq is None:
        # sum of each rows of freqmx (size = row-size)
        row_sum_vec = np.squeeze(np.asarray(freqmx.sum(axis=1))).astype(np.float64)
    else:
        row_sum_vec = np.array([nfreq[e] for e in freqMTX.row_items])
    if cfreq is None:
        # sum of each columns of freqmx (size = column-size)
        col_sum_vec = np.squeeze(np.asarray(freqmx.sum(axis=0))).astype(np.float64)
    else:
        col_sum_vec = np.array([cfreq[e] for e in freqMTX.col_items])

    freqmx = freqmx.tocoo()
    row = freqmx.row  # indices of row
    col = freqmx.col  # indices of column
    data = freqmx.data
    # o11 / e11 -> o11 / ((R1 * C1) / N) -> o11 * N / (R1 * C1)
    data = data * tot_sum  # -> o11

    ppmi = np.zeros(freqmx.nnz, dtype=np.float32)
    for i in progbar(range(freqmx.nnz)):
        rowid, colid = row[i], col[i]
        e11 = row_sum_vec[rowid] * col_sum_vec[colid]
        x = (data[i] / e11) if e11 > 0 else 0
        pmi = math.log(x if x != 0 else 1.0)
        if positive:
            ppmi[i] = 0.0 if pmi < 0 else pmi
        else:
            ppmi[i] = pmi

    ppmimx = sp.csr_matrix((ppmi, (row, col)), shape=freqmx.shape)
    args = (ppmimx, freqMTX.row_items, freqMTX.col_items)
    kwargs = {
        'category': 'association', 'meas': 'ppmi',
    }
    return freqMTX.__class__(*args, **kwargs)


@timeit
def compute_association(freqMTX, nfreq, cfreq, N=None, meas='ppmi'):
    """Compute association measures matrix.
    
    The matrix provided can be a submatrix with selected rows and/or columns, but `nfreq`
    and `cfreq` must be marginal frequencies from a reference matrix, i.e. with co-occurrence
    frequencies for the full corpus. `N` should be the sum of that reference matrix:
    if it is not provided, it will be computed as the sum of row or column marginal frequencies
    (whatever is largest).

    Parameters
    ----------
    freqMTX : :class:`~qlvl.TypeTokenMatrix`
        Raw co-occurrence frequency matrix.
    nfreq : :class:`~qlvl.Vocab`
        Marginal row frequencies of the reference matrix.
    cfreq : :class:`~qlvl.Vocab`
        Marginal collocate frequencies of the reference matrix.
    N : int
        Sum of the reference frequency matrix.
    meas : str
        Implemented association measures: 'pmi', 'ppmi', 'llik' (log likelihood),
        'chisq', 'zscore', 'dice'.

    Returns
    -------
    association measure matrix : :class:`~qlvl.TypeTokenMatrix`
    """
    N = N if N else max(nfreq.sum(), cfreq.sum())
    nfreq = np.array([nfreq[e] for e in freqMTX.row_items])
    cfreq = np.array([cfreq[e] for e in freqMTX.col_items])

    measmx = calc_association(freqMTX.matrix, nfreq=nfreq, cfreq=cfreq, N=N, meas=meas)
    args = (measmx, freqMTX.row_items, freqMTX.col_items)
    kwargs = {
        'category': 'association', 'meas': meas,
    }
    return freqMTX.__class__(*args, **kwargs)


def calc_association(freqmx, nfreq=None, cfreq=None, N=None, meas='ppmi'):
    """Compute association measures matrix.

    Parameters
    ----------
    freqmx : scipy.sparse.csr_matrix
        Raw co-occurrence frequency matrix.
    nfreq : list or numpy.ndarray
        Node frequency (sum).
    cfreq : list or numpy.ndarray
        Collocate frequency (sum).
    N : int or float
        Total frequency.
    meas : str
        Implemented association measures: 'pmi', 'ppmi', 'llik' (log likelihood),
        'chisq', 'zscore', 'dice'.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    #               Collocate present   Collocate absent    Totals
    # Node present  c_a_b               c_a_nb              R1
    # Node absent   c_na_b              c_na_nb             R2
    # Totals        C1                  C2                  N

    # normally we would use the passed node frequency array
    # because they are calculated on the complete collocate frequency matrix
    # if calculated on the passed freqmx
    # we are calculating on a subset of the complete matrix
    if nfreq is None:
        # sum of each rows of freqmx (size = row-size)
        row_sum_vec = np.squeeze(np.asarray(freqmx.sum(axis=1))).astype(np.float64)
    else:
        row_sum_vec = np.array(nfreq) if isinstance(nfreq, list) else nfreq
    if cfreq is None:
        # sum of each columns of freqmx (size = column-size)
        col_sum_vec = np.squeeze(np.asarray(freqmx.sum(axis=0))).astype(np.float64)
    else:
        col_sum_vec = np.array(cfreq) if isinstance(cfreq, list) else cfreq

    # tot_sum = float(row_sum_vec.sum())
    tot_sum = N  # total frequency, row_sum_vec.sum() should be equal to col_sum_vec.sum()
    freqmx = freqmx.tocoo()  # transform to COOrdinate format, for easier usage
    row = freqmx.row  # -> stores row indices of values in csr_matrix
    col = freqmx.col  # -> stores column indices of values in csr_matrix
    data = freqmx.data  # -> stores values in csr_matrix

    func_dict = {
        'pmi': calc_pmi,
        'ppmi': calc_ppmi,
        'lik': calc_lik,
        'llik': calc_lik,
        'chisq': calc_chisq,
        'zscore': calc_zscore,
        'dice': calc_dice,
        'deltap': calc_deltap,
        'deltapColl': calc_deltap_coll,
        'logratio': calc_log_ratio,
    }
    if meas not in func_dict:
        raise NotImplementedError("Unsupported association measure: {}".format(meas))

    afunc = func_dict[meas]
    measmx = np.zeros(len(data), dtype=np.float32)
    for i in progbar(range(freqmx.nnz)):
        rowid, colid = row[i], col[i]  # row and column indices of a value
        # prepare: c_a_b, c_a_nb, c_na_b, c_na_nb
        c_a_b = data[i]
        c_a_nb = row_sum_vec[rowid] - c_a_b
        c_na_b = col_sum_vec[colid] - c_a_b
        c_na_nb = tot_sum - c_a_nb - c_na_b - c_a_b
        measmx[i] = afunc(c_a_b, c_na_b, c_a_nb, c_na_nb)

    measmx = sp.csr_matrix((measmx, (row, col)), shape=freqmx.shape)
    return measmx


# deprecated
def pairwise_association(c_a_b, c_na_b, c_a_nb, c_na_nb, meas='ppmi'):
    func_dict = {
        'pmi': calc_pmi,
        'ppmi': calc_ppmi,
        'lik': calc_lik,
        'chisq': calc_chisq,
        'zscore': calc_zscore,
        'dice': calc_dice,
        'deltap': calc_deltap,
        'deltapColl': calc_deltap_coll,
        'logratio': calc_log_ratio,
    }
    if meas not in func_dict:
        raise NotImplementedError("Unsupported association measure: {}".format(meas))

    afunc = func_dict[meas]
    args = (c_a_b, c_na_b, c_a_nb, c_na_nb)
    return afunc(*args)


def calc_pmi(c_a_b, c_na_b, c_a_nb, c_na_nb):
    """Calculate PMI value"""
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    N = o11 + o21 + o12 + o22
    '''
    jprob = o11 / N
    xprob = (o11 + o21) / N
    yprob = (o11 + o12) / N
    pmi = math.log(adjust_val(jprob / (xprob * yprob)))
    '''
    e11 = (o11 + o12) * (o11 + o21) / N
    e11 = adjust_val(e11)
    pmi = math.log(adjust_val(o11 / e11))
    return pmi


def calc_ppmi(c_a_b, c_na_b, c_a_nb, c_na_nb):
    """Calculate PPMI value"""
    pmi = calc_pmi(c_a_b, c_na_b, c_a_nb, c_na_nb)
    ppmi = 0.0 if pmi < 0.0 else pmi
    return ppmi


def calc_lik(c_a_b, c_na_b, c_a_nb, c_na_nb):
    """Calculate log-likelihood"""
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    p1 = o11 / (o11 + o21)
    p1 = adjust_val(p1)
    p2 = o12 / (o12 + o22)
    p2 = adjust_val(p2)
    p = (o11 + o12) / (o11 + o12 + o21 + o22)
    p = adjust_val(p)

    res = 2.0 * (
        o11 * math.log(p1) + o21 * math.log(1.0 - p1) +
        o12 * math.log(p2) + o22 * math.log(1.0 - p2) -
        o11 * math.log(p)  - o21 * math.log(1.0 - p)  -
        o12 * math.log(p)  - o22 * math.log(1.0 - p)
    )
    return res


def calc_chisq(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    chisquare = (
        (o11 + o21 + o12 + o22) *
        (((o11 * o22) - (o21 * o12)) ** 2) /
        ((o11 + o21) * (o11 + o12) * (o21 + o22) * (o12 + o22))
    )
    return chisquare


def calc_zscore(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    N = o11 + o21 + o12 + o22
    e11 = ((o11 + o21) * (o11 + o12)) / N  # E11 = R1 * C1 / N
    zscore = (
        (o11 - e11) / math.sqrt(e11)
    )
    return zscore


def calc_mi(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    N = o11 + o21 + o12 + o22
    e11 = ((o11 + o21) * (o11 + o12)) / N  # E11 = R1 * C1 / N
    mi = math.log(o11 / e11, 2)
    return mi


def calc_dice(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    r1 = o11 + o12
    c1 = o11 + o21
    dice = 2 * o11 / (r1 + c1)
    return dice


def calc_deltap(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    deltap = o11 / (o11 + o12) - o21 / (o21 + o22)
    return deltap


def calc_deltap_coll(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    deltap_coll = o11 / (o11 + o21) - o12 / (o12 + o22)
    return deltap_coll


def calc_log_ratio(c_a_b, c_na_b, c_a_nb, c_na_nb):
    o11, o21, o12, o22 = float(c_a_b), float(c_na_b), float(c_a_nb), float(c_na_nb)
    # ratio = (o11 * (o21 + o22)) / (o21 * (o21 + o22)) if o21 != 0 else 1.0
    ratio = (o11 / (o11 + o12)) / (o21 / (o21 + o22)) if o21 != 0 else 1.0
    logr = math.log(adjust_val(ratio))
    return logr


def entropy(values):
    """Compute the entropy of a data set"""
    e = 0.0
    for val in values:
        e -= val * math.log(adjust_val(val))
    return e


def adjust_val(val):
    """for log calculation"""
    if val == 0:
        val = 0.0000000001
    elif val == 1:
        val = 0.9999999999
    return val


@timeit
def compute_distance(measMTX, axis=0, metric='cosine'):
    """Compute distance matrix from association measure matrix

    Parameters
    ----------
    measMTX : :class:`~qlvl.TypeTokenMatrix`
    axis : int
        0 (row) or 1 (column)
    metric : str
        'cosine' (default), 'euclidean', 'cityblock', 'l1', 'l2', 'manhattan'
        metrics that are valid in *sklearn.metrics.pairwise_distances*

    Returns
    -------
    :class:`~qlvl.TypeTokenMatrix`

    """
    if axis == 0:
        measmx = measMTX.matrix
        items = measMTX.row_items  # row-by-row (word-by-word) distance matrix
    elif axis == 1:
        measmx = measMTX.matrix.transpose()
        items = measMTX.col_items  # column-by-column (context-by-context) distance matrix
    else:
        raise ValueError("Axis should be 0 or 1!")
    if isinstance(measmx, sp.spmatrix):
        measmx = measmx.toarray()  # faster if transform to numpy.ndarray first
    dtypes = {
        'cosine': 'cos',
        'cos': 'cos',
        'euclidean': 'euclid',
        'cityblock': 'cb',
        'manhattan': 'manh',
        'l1': 'l1',
        'l2': 'l2',
    }
    if metric not in dtypes:
        raise NotImplementedError("Unknown metric {}".format(metric))
    distmx = calc_distance(measmx, metric=metric)

    kwargs = {
        'matrix': distmx, 'row_items': items, 'col_items': items,  # necessary parameters
        'category': 'distance', 'metric': dtypes[metric],          # unnecessary, just for info
    }
    return measMTX.__class__(**kwargs)


def calc_distance(measmx, metric='cosine'):
    """Call sklearn.metrics.pairwise_distances function to calculate distance matrix."""
    distmx = pairwise_distances(measmx, metric=metric)
    return distmx


def compute_cos(measMTX, axis=0):
    return compute_cosine(measMTX, axis=axis)


@timeit
def compute_cosine(measMTX, axis=0):
    """

    Parameters
    ----------
    measMTX : :class:`~qlvl.TypeTokenMatrix`

    """
    if axis == 0:
        measmx = measMTX.matrix
        items = measMTX.row_items
    elif axis == 1:
        measmx = measMTX.matrix.transpose()
        items = measMTX.col_items
    else:
        raise ValueError("Axis should be 0 or 1!")
    if isinstance(measmx, sp.spmatrix):
        measmx = measmx.toarray()

    # sklearn.metrics.pairwise.cosine_similarity returns a symmetric numpy.ndarray
    cosmx = cosine_similarity(measmx)
    # convert diagonal values into real ones
    # TODO: check if there are other ways to correct this error?
    for i in range(cosmx.shape[0]):
        cosmx[i, i] = 1.0

    args = (cosmx, items, items)
    kwargs = {'category': 'similarity', 'metric': 'cos'}
    return measMTX.__class__(*args, **kwargs)


@timeit
def compute_simrank(simMTX, reverse=False):
    """Compute similarity rank matrix.

    Parameters
    ----------
    simMTX : cosine similarity SquareMatrix
    reverse : boolean, default False
        True: the rank 1 represents the most similar one of that row
        False the largest rank 1 represents the most similar one of that row
    """
    cosmx = simMTX.matrix
    if reverse:
        idx_mtx = np.argsort(cosmx, axis=1)
    else:
        idx_mtx = np.argsort(-cosmx, axis=1)
    m, n = idx_mtx.shape[0], idx_mtx.shape[1]

    x = np.array([i for i in range(1, n + 1)])
    y = np.array([i for i in range(m)])
    I, J = np.meshgrid(x, y)
    rJ, rInd = J.ravel(), idx_mtx.ravel()

    simrank_mtx = np.zeros((m, n), int)
    simrank_mtx[rJ, rInd] = I.ravel()

    metric = simMTX.metric if 'metric' in simMTX.__dict__ else 'cos'
    args = (simrank_mtx, simMTX.row_items, simMTX.col_items)
    kwargs = {'category': 'rank', 'metric': metric}
    return simMTX.__class__(*args, **kwargs)


def compute_token_weights(tcPositionMTX, twMTX, booleanize = True, tokenFormat='lemma/pos'):
    """Compute token-by-context weight matrix.
    Build token weights from a token-by-context matrix and
    a type-by-context weight matrix.

    Parameters
    ----------
    tcPositionMTX : :class:`~qlvl.TypeTokenMatrix`
        token-by-context position matrix
                               target words
                             ---------------
                            |               |
                     tokens |               |
                            |               |
                             ---------------
    twMTX : :class:`~qlvl.TypeTokenMatrix`
        type-by-context weight matrix, i.e. 'ppmi' (transposed)
                               target words
                             ---------------
                    context |      ...      |
                   features | ...   x   ... |
                    (types) |      ...      |
                             ---------------

    booleanize : bool
        whether `tcPositionMTX` should be booleanized. If False, original values will be kept.
    
    tokenFormat : str
        whether settings['token'] has both lemma and part-of-speech information (default) or just lemma information
    
    Returns
    -------
    token weight matrix : :class:`~qlvl.TypeTokenMatrix`

    Notes
    -----
    This function will transform all explicit zeros in type-by-context weight matrix to implicit zeros.
    So, if those explicit zeros a important, be careful of them.
    """
    # check the column items of tcPositionMTX and twMTX are in the same order
    col_items1, col_items2 = tcPositionMTX.col_items, twMTX.col_items
    if col_items1 != col_items2:
        raise ValueError("Columns of the two matrices are not in the same order ")

    missing_types = []
    tokens = tcPositionMTX.row_items
    types = set(twMTX.row_items)  # set of target types
    # tcmx_type = np.bool if booleanize else tcPositionMTX.matrix.dtype
    # Adding _ to make sure it works with numpy 1.26.4 in Google Colab
    tcmx_type = np.bool_ if booleanize else tcPositionMTX.matrix.dtype
    bool_tcmx = tcPositionMTX.matrix.astype(tcmx_type, copy=True).toarray()
    twmx = twMTX.matrix.toarray()  # transform to dense matrix (numpy.ndarray)
    resmx = np.zeros(bool_tcmx.shape)
    for i, tok in enumerate(tokens):
        # normally the last two parts of token string are filename and line number
        # while the first one or two parts of token string are 'lemma' or 'lemma/pos'
        # so split token string by '/' from right twice -> 'lemma' or 'lemma/pos', 'fname', 'lid'
        # type_ = '/'.join(tok.rsplit('/', 2)[:-2])
        # change 2023.04.11: flexible softcoding of token id (given corpora with different structure)
        type_ = '/'.join(tok.split('/')[:2]) if tokenFormat == 'lemma/pos' else tok.split('/')[0]
        if type_ not in types:
            missing_types.append(type_)
            continue
        idx = twMTX.row_items.index(type_)
        resmx[i] = bool_tcmx[i] * twmx[idx]  # use boolean masks to select weights of corresponding indices

    if len(missing_types) > 0:
        logger.warning("Missing types in type-weight matrix:\n{}...".format(str(missing_types[:7])))
    tok_weight_mtx = sp.csr_matrix(resmx)
    return tcPositionMTX.__class__(tok_weight_mtx, tcPositionMTX.row_items, tcPositionMTX.col_items)


def compute_token_vectors(tcWeightMTX, soccMTX, operation='addition', normalization='l1'): # by Stefano
    """Compute token vectors.
    Build token vectors from a token weights (token-by-context weight matrix) and
    a second order matrix.

    Parameters
    ----------
    tcWeightMTX : :class:`~qlvl.TypeTokenMatrix`
        Token-Context weight matrix.
    soccMTX : :class:`~qlvl.TypeTokenMatrix`
        Second order collocate matrix.
    operation : str
        'addition', 'multiplication','weightedmean'
    normalization: str
        'l1', 'l2', 'no'

    Returns
    -------
    token vectors : :class:`~qlvl.TypeTokenMatrix`
    
    Note
    -----
    Values for "normalization" are regulated by sklearn.preprocessing.normalize()
    """
    # check pre-requisites
    # 1. Matrix type
    # assert isinstance(tcWeightMTX, TypeTokenMatrix)
    # assert isinstance(soccMTX, TypeTokenMatrix)
    assert tcWeightMTX.__class__ == soccMTX.__class__
    # 2. check item list consistency
    inter_items_1 = tcWeightMTX.col_items
    inter_items_2 = soccMTX.row_items
    isequalset = set(inter_items_1) == set(inter_items_2)
    isequallist = inter_items_1 == inter_items_2
    # 3. get a synchronized/reordered matrix from soccMTX according to (column of) tcWeightMTX
    if isequalset:
        if isequallist:
            # they are exactly the same item list
            right_mtx = soccMTX.get_matrix()
        else:
            # they have the same intermediate item set
            # but the item lists may have different order
            rightMTX = soccMTX.reorder(tcWeightMTX.col_items)
            right_mtx = rightMTX.get_matrix()
            # NOTE: we could reorder soccMTX here, because the reorder() method returns a new Matrix
    else:
        raise ValueError("Provided second order collocate matrix inconsistent with token-context weight matrix!")

    left_mtx = tcWeightMTX.get_matrix()

    if operation == 'addition':
        logger.info("  Operation: addition 'token-feature weight matrix' X 'socc matrix'...")
        product_mtx = dot_addition(left_mtx, right_mtx)
    elif operation == 'weightedmean': # addition by Stefano (2021.02.09)
        logger.info("  Operation: weighted mean 'token-feature weight matrix' X 'socc matrix'...")
        left_mtx_weightsum = left_mtx.sum(axis=1)
        weightedMTX = left_mtx.dot(right_mtx)/left_mtx_weightsum
        product_mtx = np.nan_to_num(weightedMTX)
    elif operation == 'multiplication':
        logger.info("  Operation: multiplication...")
        product_mtx = dot_multiplication(left_mtx, right_mtx)
    else:
        raise ValueError("Operation must be 'addition', 'multiplication' or 'weightedmean'.")

    product_mtx = sp.csr_matrix(product_mtx)
    if normalization != 'no': # addition by Stefano 2021.02.09, adapted by Mariana 2021.08.20
        product_mtx = preprocessing.normalize(product_mtx, norm=normalization)
    return tcWeightMTX.__class__(product_mtx, tcWeightMTX.row_items, soccMTX.col_items)


def dot_addition(mtx_l, mtx_r):
    """Dot addition of two matrices"""
    # check mtx type
    assert isinstance(mtx_l, sp.csr_matrix)
    assert isinstance(mtx_r, sp.csr_matrix)
    return mtx_l.dot(mtx_r).toarray()


def dot_multiplication(mtx_l, mtx_r):
    """Dot multiplication of two matrices"""
    m = mtx_l.shape[0]
    n = mtx_r.shape[1]
    product_mtx = np.ones((m, n), dtype=np.float64)
    for i in range(m):
        for k in range(n):
            for j in range(mtx_l.shape[1]):
                if mtx_l[i, j] == 0.0 or mtx_r[j, k]:
                    continue
                product_mtx[i, k] *= mtx_l[i, j] * mtx_r[j, k]
    return product_mtx
