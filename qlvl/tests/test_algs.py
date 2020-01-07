"""
Test algorithm functions
"""

import os
from collections import Counter

import numpy as np
import pytest
import random
import scipy.cluster.hierarchy as sch

from qlvl.models import cbc

curdir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def X():
    X = np.array(
        [[5, 3],
         [10, 15],
         [15, 12],
         [24, 10],
         [30, 30],
         [85, 70],
         [71, 80],
         [60, 78],
         [70, 55],
         [80, 91]]
    )
    '''
    Hierarchical Clustering Dendrogram ('single')
        ---------------
        |             |
     -------        ------
     |     |        |    |
     |     |        |  -----
     |     |        |  |   |
     |     |        |  |  _____     
     |  ------      |  |  |   |
     |  |    |      |  |  |  ----
     |  |  -----    |  |  |  |  |
     |  |  |   |    |  |  |  |  |
     |  |  |  ----  |  |  |  |  |
     |  |  |  |  |  |  |  |  |  |
    ------------------------------
     5  1  4  2  3  9  6  10 7  8
    '''
    yield X


class TestAlgs(object):
    def test_flat_clusters(self, X):
        Z = sch.linkage(X, 'single')
        # np.array([v for _1, _2, v, _4 in Z])  ->  'merge' distances
        # [ 5.83095189  9.21954446 11.18033989 13.         14.2126704  17.20465053  20.88061302 21.21320344 47.16990566]
        # test criterion = 'distance'
        clusters = cbc.flat_cluster(Z, t='median', criterion='distance')
        assert np.array_equal(clusters, np.array([1, 1, 1, 1, 2, 4, 3, 3, 5, 3]))
        clusters = cbc.flat_cluster(Z, t='25%', criterion='distance')
        assert np.array_equal(clusters, np.array([2, 1, 1, 1, 3, 6, 4, 4, 7, 5]))
        clusters = cbc.flat_cluster(Z, t='75%', criterion='distance')
        assert np.array_equal(clusters, np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 2]))
        clusters = cbc.flat_cluster(Z, t='mean', criterion='distance')
        assert np.array_equal(clusters, np.array([1, 1, 1, 1, 2, 3, 3, 3, 4, 3]))
        # test criterion = 'minsize'
        minsize = 3
        clusters = cbc.flat_cluster(Z, t=minsize, criterion='minsize')
        clusters = Counter(clusters)
        assert minsize in clusters.values()

    def test_score_cluster(self):
        pass

    def test_average_similarity(self):
        # 4-element similarity matrix
        simarr = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        assert cbc.average_similarity([1, 2], simarr) == 0.3
        assert np.isclose(cbc.average_similarity([1, 2, 3], simarr), 0.2)

    def test_cluster_avgsim(self):
        # 4-element similarity matrix
        simarr = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        clusters = {
            1: [1],
            2: [0, 2, 3],
        }
        cluster_sims = cbc.cluster_avgsim(clusters, simarr)

    def test_sub2glb(self):
        for _ in range(10):
            glbitems = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k']
            subsize = random.randint(1, len(glbitems))  # -> size of sub
            subidx = random.sample(range(len(glbitems)), subsize)  # randomly select 'subsize' values from target list
            subitems = [glbitems[i] for i in subidx]

            csize = random.randint(1, len(subidx))  # random cluster size
            cluster = random.sample(range(len(subidx)), csize)  # random cluster
            glbidx = [subidx[i] for i in cluster]
            glbclust = cbc.sub2glb(cluster, subitems, glbitems)
            assert glbclust == glbidx

    def test_cbc_step1(self):
        pass

    def test_cbc_step2(self):
        L = [(0.5, [1, 2, 3]), (0.3, [0, 3, 5]), (0.1, [3, 4, 6]), (0.4, [2, 4, 5]), (0.2, [3, 5, 6])]
        sortedL = cbc.cbc_step2(L)
        assert [0.5, 0.4, 0.3, 0.2, 0.1] == [maxv for maxv, _ in sortedL]

    def test_remove_duplicate(self):
        L = [(0.5, [1, 2, 3]), (0.4, [2, 4, 5]), (0.40000000004, [2, 5, 4]), (0.3, [0, 3, 5]), (0.2, [3, 5, 6]), (0.1, [3, 4, 6])]
        diffL = cbc.remove_duplicate(L)
        assert [v for v, c in diffL] == [0.5, 0.4, 0.3, 0.2, 0.1]

    def test_cbc_step3(self):
        pass

    def test_calc_meas_vec(self):
        pass
