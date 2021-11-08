"""
Test TypeTokenMatrix Calculation functions
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.spatial.distance import squareform

from nephosem import Vocab, TypeTokenMatrix
from nephosem.specutils import mxcalc

curdir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def df():
    # node, colloc, co-occurrences, node count, colloc count, total count, PMI, PPMI, Log-likelihood
    df = pd.read_csv("{}/data/toy.values.csv".format(curdir), sep='\t')
    yield df


@pytest.fixture()
def nfreq():
    '''
    node_freq = {
        '.': 10, 'and': 9, 'at': 7, 'bark': 7, 'be': 15, 'better': 5,
        'by': 18, 'cat': 15, 'coffee': 14, 'cup': 9, 'dog': 15,
        'grab': 27, 'his': 18, 'it': 10, 'its': 8, 'loud': 12,
        'neck': 8, 'passerby': 7, 'purr': 5, 'scratch': 10, 'taste': 14,
        'tea': 5, 'than': 5, 'the': 64, 'vet': 27, 'while': 10
    }
    nfreq = Vocab(node_freq)
    '''
    nfreq = Vocab.load("{}/data/toy.nfreq".format(curdir), fmt='plain')
    yield nfreq


@pytest.fixture()
def cfreq():
    '''
    colloc_freq = {
        '.': 44, 'and': 8, 'at': 6, 'bark': 6, 'be': 14, 'better': 4,
        'by': 17, 'cat': 14, 'coffee': 12, 'cup': 8, 'dog': 13,
        'grab': 25, 'his': 16, 'it': 10, 'its': 7, 'loud': 10,
        'neck': 7, 'passerby': 6, 'purr': 4, 'scratch': 10, 'taste': 12,
        'tea': 4, 'than': 4, 'the': 58, 'vet': 25, 'while': 10
    }
    cfreq = Vocab(colloc_freq)
    '''
    cfreq = Vocab.load("{}/data/toy.cfreq".format(curdir), fmt='plain')
    yield cfreq


@pytest.fixture()
def spMTX():
    # this is a sparse matrix (scipy.sparse.csr_matrix)
    sparr = np.array([
        [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 1]
    ])
    spmx = sp.csr_matrix(sparr)
    row_items = ['dog', 'cat', 'coffee']
    col_items = ['bark', 'passerby', 'vet', 'grab', 'neck', 'purr',
                 'loud', 'scratch', 'taste', 'better', 'tea', 'cup']
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


class TestCalcFuncs(object):
    def test_compute_ppmi(self, spMTX, nfreq, cfreq, df):
        # compare pmi values
        pmiMTX = mxcalc.compute_ppmi(spMTX, nfreq=nfreq, cfreq=cfreq, positive=False)
        for i in range(df.shape[0]):
            row = df.iloc[i]
            node, colloc = row['node'], row['colloc']
            assert abs(pmiMTX[node, colloc] - row['pmi']) < 0.0000001

        # compare ppmi values
        ppmiMTX = mxcalc.compute_ppmi(spMTX, nfreq=nfreq, cfreq=cfreq, positive=True)
        for i in range(df.shape[0]):
            row = df.iloc[i]
            node, colloc = row['node'], row['colloc']
            assert abs(ppmiMTX[node, colloc] - row['ppmi']) < 0.0000001

    def test_compute_association(self, spMTX, nfreq, cfreq, df):
        pmiMTX = mxcalc.compute_association(spMTX, nfreq=nfreq, cfreq=cfreq, meas='pmi')
        # compare pmi values
        for i in range(df.shape[0]):
            row = df.iloc[i]
            node, colloc = row['node'], row['colloc']
            assert abs(pmiMTX[node, colloc] - row['pmi']) < 0.0000001

        # compare ppmi values
        ppmiMTX = mxcalc.compute_association(spMTX, nfreq=nfreq, cfreq=cfreq, meas='ppmi')
        for i in range(df.shape[0]):
            row = df.iloc[i]
            node, colloc = row['node'], row['colloc']
            assert abs(ppmiMTX[node, colloc] - row['ppmi']) < 0.0000001

        # compare log-likelihood values
        ppmiMTX = mxcalc.compute_association(spMTX, nfreq=nfreq, cfreq=cfreq, meas='lik')
        for i in range(df.shape[0]):
            row = df.iloc[i]
            node, colloc = row['node'], row['colloc']
            assert abs(ppmiMTX[node, colloc] - row['lik']) < 0.0000001

    def test_calc_pmi(self):
        # case 1
        o11 = 1159
        R1 = 1938
        C1 = 1311
        N = 50000952
        e11 = (R1 * C1) / N
        print(math.log(o11 / e11))

        N = 50000952
        # cases: R1, C1, O11
        cases = [(1938, 1311, 1159),
                 (5578, 2749, 1384),
                 (283891, 3293296, 3347),
                 (1761436, 1375396, 1190)]
        pmis = [10.0349081703, 8.41470768304, -1.72037278119, -3.70663100173]
        # transform to (c_a_b, c_na_b, c_a_nb, c_na_nb)
        cases = [(o11, C1 - o11, R1 - o11, N - R1 - C1 + o11) for R1, C1, o11 in cases]
        for c, p in zip(cases, pmis):
            pmi = mxcalc.calc_pmi(c[0], c[1], c[2], c[3])
            assert abs(pmi - p) < 0.00000001
