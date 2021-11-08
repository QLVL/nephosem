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


"""Algorithms

Usage examples
==============

Initialize a vocabulary with a Python dict e.g.

>>> from nephosem.algos import cbc

"""

import codecs
import datetime
import json
import logging
import math
import operator
import os
from collections import defaultdict
from copy import deepcopy
from multiprocessing import cpu_count, Pool

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from nephosem import progbar
from nephosem.core.matrix import TypeTokenMatrix
from nephosem.specutils.mxcalc import calc_association, calc_distance
from nephosem.specutils.mxutils import centroid_of_cluster, sum_of_cluster
from nephosem.utils import make_dir, clean_dir

__all__ = ['CBC']

logger = logging.getLogger(__name__)


class CBC(object):

    def __init__(self, elements, freqmx, measmx=None, distmx=None,
                 k=100, theta1=0.35, theta2=0.25,
                 prune_method='distance', t='median',
                 score_metric='without_size', highest_score=False,
                 num_iter=1000, workers=-1):
        """Cluster by Committee.

        Parameters
        ----------
        elements : iterable
            a list of elements/items (str)
        freqmx : scipy.sparse.csr_matrix
            Collocate frequency matrix
        measmx : scipy.sparse.csr_matrix, optional
            Association measure matrix based on freqmx.
        distmx : numpy.ndarray, optional
            Distance matrix on rows of freqmx.
        k : int, optional
            Number of similar elements to cluster for each element
            Default 10.
        theta1 : float
            Parameter theta one, default 0.35.
        theta2 : float
            Parameter theta two, default 0.25.
        prune_method : str
            Method to prune the hierarchical clustering tree.
            ``distance`` :
                Prune the tree by drawing a horizontal line on height (distance) equaling to *t* (threshold).
            ``minsize`` :
                Prune the tree by drawing a horizontal line on a minimum height
                when there is one cluster containing more than (or equal to) *t* elements among clusters below this line.
        t : int or float or str, optional
            The threshold to apply when pruning clustering tree.
        score_metric : str, optional
            Specify the metric to calculate the score of a cluster.
            ``without_size`` :
                Use formula : avgsim(c).
                Scores a cluster by only its average similarity.
            ``with_size`` :
                Use formula : |c| * avgsim(c).
                Scores a cluster by the product of its size and its average similarity.
            ``with_size_sqrt`` :
                Use formula : sqrt(|c|) * avgsim(c).
                Scores a cluster by the product of the sqrt of its size and its average similarity.
        highest_score : bool, optional
            If True, retrieve only the highest scoring cluster based on scoring metric.
            Else, retrieve all clusters by ``hierarchical_cluster`` function.
        num_iter : integer
            Number of iterations, default 1,000.
            If the number of iterations is larger than `num_iter`, then the algorithm stops.
        workers

        Returns
        -------
        Cs : list
            A list of committees of each recursions.
        Rs : list
            A list of residues of each recursions.
        """

        # prepare matrices if not provided
        if measmx is None:
            print("calculating ppmi matrix...")
            measmx = calc_association(freqmx, meas='ppmi')
            print("done.")
        if distmx is None:
            print("calculating distance matrix...")
            distmx = calc_distance(measmx)
            print("done.")

        self.elements = elements
        self.freqmx = freqmx.copy()
        self.measmx = measmx.copy()
        self.distmx = distmx.copy()

        self.k = k
        self.theta1 = theta1
        self.theta2 = theta2
        self.prune_method = prune_method
        self.t = t
        self.score_metric = score_metric
        self.highest_score = highest_score
        self.num_iter = num_iter
        self.workers = workers if workers > 0 else cpu_count() - 1

    def cluster(self, num_eles=-1, multicore=True):
        """Main method of Cluster by Committee.

        Parameters
        ----------
        num_eles : int, optional
            Cluster only the first `num_eles` elements.
            ``Notes`` : Normally for test.
        multicore : bool, optional
            Use multicore method or not.
            Default True.

        Returns
        -------
        (Cs, Rs) : tuple
            A list of committees and a list of residues (of each recursion).
        """
        # give description of CBC clustering
        logger.info("Clustering {} elements for top-{} similar neighbors with\n"
                    "'{}' pruning method and '{}' threshold\n"
                    "'{}' scoring metric and return {}"
                    .format(num_eles if num_eles > 0 else len(self.elements), self.k,
                            self.prune_method, self.t,
                            self.score_metric,
                            'highest scoring cluster' if self.highest_score else 'all clusters'))

        # process only a number of elements or the whole
        residues = self.elements[:num_eles] if num_eles > 0 else self.elements[:]
        Cs, Rs = [], []  # -> a list of committees and residues of each recursion

        # create tmp folder for multicore methods
        tmpdir = os.path.join(os.path.expanduser('~'), '.cbc')
        make_dir(tmpdir)

        i = 1
        while i <= self.num_iter and len(residues) > 0:
            separator = '-' * 33
            logger.info("{}\nRECURSION {}: {} elements".format(separator, i, len(residues)))
            try:
                curC, curR = major_steps(self, eles=residues, multicore=multicore)
            except Exception as err:
                logger.exception(err)
                curC, curR = [], []

            Cs.append(curC)
            Rs.append(curR)

            # stop when there is only a few residues left
            if len(curR) <= 2:
                logger.info("Less than three residues left! Just return!")
                return Cs, Rs
            # if the residues got from this recursion are the same as elements of this recursion
            # then no new committees found in this recursion
            if len(curR) == len(residues):
                logger.info("No new committee anymore!")
                return Cs, Rs
            residues = curR

            i += 1

        restmpdir = os.path.join(tmpdir, 'res.{}'.format(datetime.datetime.now().strftime("%Y.%m.%d.%H.%M")))
        make_dir(restmpdir)
        try:
            save_committees_json(restmpdir, Cs)
            save_residues_json(restmpdir, Rs)
        except Exception as err:
            logger.exception(err)

        return Cs, Rs


def major_steps(cbc, eles=None, multicore=True):
    """Major steps of modification of phase II of Cluster by Committee.
    Step 1: cluster top-k similar elements of each target element.
    Step 2: clean clusters and merge similar ones.
    Step 3: for each element, if it is not similar enough to all clusters, keep it for later recursions
            else, merge it to the most similar cluster.

    Parameters
    ----------
    cbc : :class:`~nephosem.CBC`
    eles : iterable, optional
        A list of elements to be clustered.
        In later recursions, `eles` are residues.
    multicore

    Returns
    -------
    (committees, residues) : tuple
        A list of committees and residues of current recursion.
    """
    if eles is None:  # if not passed, use
        eles = cbc.elements
    # in later recursions use submatrix of residues to speed up
    # and we have to, because the similar elements of a residue are fetched from all residues
    freqmx = cbc.freqmx.submatrix(row=eles) if cbc.elements != eles else cbc.freqmx
    measmx = cbc.measmx.submatrix(row=eles) if cbc.elements != eles else cbc.measmx
    distmx = cbc.distmx.submatrix(row=eles, col=eles) if cbc.elements != eles else cbc.distmx

    # step 1
    logger.info("\nstep 1 ...")
    num_workers = cpu_count() - 1
    args = (eles,)
    kwargs = {
        'distmx': distmx, 'k': cbc.k,
        'prune_method': cbc.prune_method, 't': cbc.t,
        'score_metric': cbc.score_metric,
    }
    if multicore and len(eles) > num_workers:
        # use multicore version when the number of elements is large ( > number of cpu cores)
        L = cbc_step1_multicore(*args, **kwargs)
    else:
        L = cbc_step1_single(*args, **kwargs)
    logger.info("get {} clusters".format(len(L)))

    # step 2
    logger.info("\nstep 2 ...")
    curC, commx = cbc_step2(L, elements=eles, freqmx=freqmx.matrix.toarray(), ppmimx=measmx.matrix.toarray(), theta=cbc.theta1)
    if len(curC) == 0:
        return [], eles
    logger.info("after reducing, remain {} clusters".format(len(curC)))

    # item2id = {e: i for i, e in enumerate(freqmx.row_items)}
    # clusters = [(v, [item2id[e] for e in c]) for v, c in L]
    assert len(curC) == commx.shape[0]

    # step 3
    logger.info("step 3 ...")
    # Only in first recursion, the elements are the elements to be clustered
    # in later recursions, the elements are residues of last recursion
    ppmimx = measmx.matrix.toarray()
    R, comms = cbc_step3(eles, measmx=ppmimx, comms=curC, commx=commx, theta=cbc.theta2)
    logger.info("number of residues: {}".format(len(R)))

    # NOTE: the returned current Cluster should be items not ids
    #       because the passed in matrices would be sub-matrices
    #       so the ids would be sub-matrix ids
    # id2item = freqmx.row_items
    # curC = [[id2item[i] for i in c] for c in curC]
    return curC, R


def cbc_step1_multicore(elements, distmx=None, k=100,
                        prune_method='distance', t='median',
                        score_metric='without_size', highest_score=False):
    """
    ~/.cbc/.10933/
                 /L/  : folder for temporary L(s)
                 /distmx.meta
                 /distmx.npy
    """
    homedir = os.path.expanduser('~')
    tmpdir = os.path.join(homedir, '.cbc', '{}'.format(datetime.datetime.now().strftime("%Y.%m.%d.%H.%M")))
    make_dir(tmpdir)
    clean_dir(tmpdir)
    L_tmpdir = os.path.join(tmpdir, 'L')
    make_dir(L_tmpdir)
    clean_dir(L_tmpdir)
    distmx_fname = os.path.join(tmpdir, 'distmx')
    distmx.save(distmx_fname, pack=False)

    num_cores = cpu_count() - 1
    num_eles = len(elements)
    data_group = [[] for _ in range(num_cores)]
    for i in range(num_eles):
        idx = i % num_cores
        data_group[idx].append(elements[i])

    pool = Pool(processes=num_cores)
    for i in range(num_cores):
        args = (data_group[i], tmpdir,)
        kwargs = {
            'k': k,
            'prune_method': prune_method, 't': t,
            'score_metric': score_metric, 'highest_score': highest_score,
        }
        try:
            pool.apply_async(cbc_step1_call, args=args, kwds=kwargs)
        except Exception as err:
            logger.exception("Error:", err)
    pool.close()
    pool.join()

    # merge results
    L = []
    for fname in os.listdir(L_tmpdir):
        fname = os.path.join(L_tmpdir, fname)
        # load json results of sub-processes L(s)
        with codecs.open(fname, 'r') as inf:
            procL = json.load(inf)
        L.extend(procL)
        os.remove(fname)
    try:
        os.remove(distmx_fname + '.meta')
        os.remove(distmx_fname + '.npy')
        os.rmdir(L_tmpdir)
        os.rmdir(tmpdir)
    except Exception as err:
        logger.exception("clean tmp folder error: {}".format(err))
    return L


def cbc_step1_call(elements, tmpdir_proc, k=100,
                   prune_method='distance', t='median',
                   score_metric='without_size', highest_score=False):
    pid = os.getpid()
    # print(pid, len(elements), tmpdir_proc, k, prune_method, t, score_metric, highest_score)
    # load distance matrix inside tmp directory
    tmp_distmx_fname = os.path.join(tmpdir_proc, 'distmx')
    try:
        distmx = TypeTokenMatrix.load(tmp_distmx_fname, pack=False)
    except Exception as err:
        logger.exception("Matrix load error: ".format(err))
        return []
    logger.info("Starting subprocess {}...".format(pid))

    L = []
    if k < 1:
        k = int(k * distmx.shape[0])

    for e in progbar(elements):
        try:
            args = (e,)
            kwargs = {
                'distmx': distmx, 'k': k,
                'prune_method': prune_method, 't': t,
                'score_metric': score_metric, 'highest_score': highest_score,
            }
            clusters = cluster_similar_elements(*args, **kwargs)
            # print(clusters)
        except Exception as err:
            clusters = []
            logger.exception("Cannot find highest scoring cluster for element: {}\nError: {}".format(e, err))
        L.extend(clusters)

    tmp_proc_fname = os.path.join(tmpdir_proc, 'L', '{}'.format(pid))
    with codecs.open(tmp_proc_fname, 'w') as outf:
        json.dump(L, outf, ensure_ascii=False, indent=4)

    return L


def cbc_step1_single(elements, distmx=None, k=100, prune_method='distance', t='median',
                     score_metric='without_size', highest_score=False, score_func=None):
    """Get a list of highest-scoring clusters for each element.
    For each element e, cluster the top similar elements of e from S using average-lin clustering.
    For each discovered cluster c, compute the |c| x avgsim(c) score.
    Store the highest-scoring cluster in a list L.

    Parameters
    ----------
    elements : list of items (int)
    distmx : TypeTokenMatrix
    k : int or float
        int -> number of similar words to select
        float -> percentage of similar words to select
    prune_method : str
        Method to prune the hierarchical clustering tree.
        ``distance`` :
            Prune the tree by drawing a horizontal line on height (distance) equaling to *t* (threshold).
        ``minsize`` :
            Prune the tree by drawing a horizontal line on a minimum height
            when there is one cluster containing more than (or equal to) *t* elements among clusters below this line.
    t : int or float
        The threshold to apply when pruning clustering tree.
    score_metric : str
        ``without_size`` :
            Scores a cluster by only its average similarity.
        ``with_size`` :
            Scores a cluster by the product of its size and its average similarity.
        ``with_size_sqrt`` :
            Scores a cluster by the product of the sqrt of its size and its average similarity.
    highest_score : bool
        If True, retrieve only the highest scoring cluster based on scoring metric.
        Else, retrieve all clusters by ``hierarchical_cluster`` function.
    score_func : function
        A function to perform on a cluster of elements.
        Basically two parameters are considered: average similarity of the cluster and the size of it.

    """
    L = []
    if k < 1:
        k = int(k * distmx.shape[0])

    for e in progbar(elements):
        try:
            args = (e,)
            kwargs = {
                'distmx': distmx, 'k': k,
                'prune_method': prune_method, 't': t,
                'score_metric': score_metric, 'highest_score': highest_score, 'score_func': score_func,
            }
            clusters = cluster_similar_elements(*args, **kwargs)
        except Exception as err:
            clusters = []
            logger.exception("Cannot find highest scoring cluster for elements: {}\nError: {}".format(e, err))
        L.extend(clusters)

    return L


def cbc_step2(L, elements=None, freqmx=None, ppmimx=None, theta=0.35):
    """Cluster by Committee Step 2.
     - clean clusters (see `clean_clusters()`)
     - MERGE committees that are similar enough (theta1)

    Additional:
        ``Sub-Super`` :
            When confronted ith super- and sub- clusters: keep the (smaller) sub-cluster, which means keep the one
            with the highest average similarity value. In case the sub-cluster is a singleton item,
            keep the super-cluster.
            When two clusters contain some common elements: keep both clusters.
        ``PPMI`` :
            Set a limit to mutual ppmi-value to avoid the 'salt & pepper' committees
            (with high ppmi, but not synonymous at all).

    Parameters
    ----------
    L : iterable
        [ (val, cluster) ... ], cluster is a list/set of ids.
    elements : iterable of str
        A list of element strings.
    freqmx : numpy.ndarray
        Raw co-occurrence frequency matrix.
        Notes : passing scipy.spmatrix would raise error.
    ppmimx : numpy.ndarray
        Association measure (ppmi) matrix.
    theta : float, optional
        theta1 parameter in the whole CBC algorithm.

    Returns
    -------
    (mergecomms, mgsglcommx) : tuple
        A tuple of merged (with singletons) committees (element strings) and committee matrix.
    """
    if not L or len(L) <= 0:
        return L

    # clean clusters
    committees, singletons = clean_clusters(L)

    # transform cluster elements from str to int
    e2id = {e: i for i, e in enumerate(elements)}
    idxcomms = [sorted([e2id[e] for e in clust]) for s, clust in committees]

    # calculate committee matrix
    # estimated time cost: 90s (10000 X 10000 matrix)
    logger.info("calculating committee matrix...")
    commx = calc_committee_vectors(idxcomms, freqmx=freqmx)
    logger.info("done...")

    # merge two similar committees
    # calculate similarity matrix of the committee matrix
    # estimated time cost : 90s (10000 X 10000 matrix)
    simmx = cosine_similarity(commx)
    mgcomms = get_merged_committee(simmx, theta=theta)
    # merge committees
    mergecomms = []
    for i, j in mgcomms:
        lcomm = set(idxcomms[i])
        rcomm = set(idxcomms[j]) if j >= 0 else set()
        mergecomms.append(sorted(lcomm | rcomm))
    # mergecomms = [sorted(set(idxcomms[i]) | set(idxcomms[j])) for i, j in mgcomms]
    logger.info("num of merged committees: {}".format(len(mergecomms)))

    # re-calculate committee matrix
    logger.info("calculating committee matrix...")
    mgcommx = calc_committee_vectors(mergecomms, freqmx=freqmx)
    logger.info("done...")

    if len(singletons) > 0:
        # calculate similarity matrix between singletons and committees
        sglidx = sorted([e2id[e] for e in singletons])
        sglmx = ppmimx[sglidx]
        # pair-wise similarity between singletons and committees
        logger.info("calculating cosine similarity between singletons and committees..")
        sglsimmx = cosine_similarity(sglmx, mgcommx)
        logger.info("done...")

        # merge singleton with most-similar committee
        logger.info("merging singletons with committees...")
        maxidx = np.argmax(sglsimmx, axis=1)
        for i in range(len(sglidx)):
            sid = sglidx[i]
            cid = maxidx[i]
            if sglsimmx[i, cid] >= theta:
                mergecomms[cid].append(sid)
        logger.info("done...")

    mergecomms = [sorted(clust) for clust in mergecomms]
    # re-calculate committee matrix
    logger.info("calculating committee matrix...")
    mgsglcommx = calc_committee_vectors(mergecomms, freqmx=freqmx)
    logger.info("done...")
    mergecomms = [[elements[i] for i in clust] for clust in mergecomms]
    return mergecomms, mgsglcommx


def get_merged_committee(simmx, theta=0.35):
    """Find committee index pairs that could be merged, based on committee matrix.

    Parameters
    ----------
    simmx : numpy.ndarray
        Committee similarity matrix
    theta : float
        Threshold for comparing similarity

    Returns
    -------
    mergecomms : iterable of tuples
        A list of merged committee index pairs.
    """
    mergecomms = []  # -> a list of two committee indices that are merged
    merged = set()  # -> record merged committee indices

    # calculate sorted indices of similarity matrix
    sortsimmx = np.argsort(simmx)
    # merge two most-similar committees that are similar enough (theta1)
    # Notes : just once as the committee size is small
    for i in range(simmx.shape[0]):
        # if committee i has been merged previously, skip it
        if i in merged:
            continue
        # get the most similar committee of current one
        sortidx = sortsimmx[i]
        # iterate over the indices from the largest one
        idx = -1
        for j in range(-2, -(sortidx.shape[0]), -1):
            tmpidx = sortidx[j]
            if tmpidx > i and tmpidx not in merged:
                # if committee has not been processed
                # and committee has not been merged
                idx = tmpidx
                break
        # if the similarity >= theta1, merge them
        if idx < 0 or simmx[i, idx] < theta:  # not found
            mergecomms.append((i, -1))
            continue
        merged.add(idx)  # record 'most-similar' committee index
        mergecomms.append((i, idx))  # merge committees 'i' and 'most-similar'

    return mergecomms


def clean_clusters(L):
    """Clean clusters.
     - Sort L based on the first value (score of cluster) descending order.
     - Remove duplicate committees.
     - Remove supersets if subset exists.
     - Remove singletons if superset exists.

    Parameters
    ----------
    L : iterable of tuple
        Original clusters (of tuple (score, cluster))

    Returns
    -------
    cleaned clusters, singleton elements : iterable of tuple, set of str
    """
    sortedL = sorted(L, key=operator.itemgetter(0), reverse=True)
    i = 0
    while i < len(sortedL) and sortedL[i][0] > 0:
        i += 1
    clusters, singletons = sortedL[:i], sortedL[i:]
    logger.info("num of >2-eles clusters: {}, num of singletons: {}".format(len(clusters), len(singletons)))

    sgleles = set()  # -> singleton elements
    for _, c in singletons:
        if len(c) < 1:
            continue
        e = list(c)[0]
        sgleles.add(e)

    logger.info("removing duplicate >2-eles clusters...")
    # remove duplicate and transform clusters into sets
    diffL = remove_duplicate(clusters)
    logger.info("num of non-duplicate clusters: {}".format(len(diffL)))

    invidx = inverted_index(diffL)
    logger.info("removing duplicate singletons...")
    # remove singletons that have super-clusters
    sgleles = remove_singleton(sgleles, invidx)
    # singletons = [(-1, {e}) for e in sgleles]
    logger.info("num of singletons: {}".format(len(sgleles)))

    # remove clusters that have sub-clusters
    logger.info("removing super-clusters if subset exists...")
    nosuperL = remove_superset(diffL, invidx)
    logger.info("num of non-super clusters: {}".format(len(nosuperL)))

    return nosuperL, sgleles


def remove_duplicate(clusters):
    """Remove duplicate clusters"""
    if len(clusters) <= 0:
        return clusters
    prevs, prevc = clusters[0]  # -> previous score and cluster
    diffL = [(prevs, set(prevc))]
    for i in range(1, len(clusters)):
        s, clust = clusters[i]
        if abs(s - prevs) > 0.000000001 or sorted(clust) != sorted(prevc):
            # if scores of two clusters are not close => they are not the same cluster
            # if scores are very close => compare cluster elements
            diffL.append((s, set(clust)))
            prevs, prevc = s, clust
        else:  # same cluster
            continue
    return diffL


def remove_singleton(singles, invidx):
    """Remove singleton clusters that have super-clusters"""
    sgls = set()
    for e in singles:
        if e not in invidx:
            # if inverted index has such key element
            # then there must be at least one cluster containing this singleton element
            sgls.add(e)
    return sgls


def remove_superset(clusters, invidx):
    """Remove super-clusters."""
    superclusters = set()
    for cid, (_, clust) in enumerate(clusters):
        # if len(clust) == 2:  # 2-ele clusters have no sub-cluster
        #    continue
        commclust = None
        # common cluster: clusters containing elements in this clust
        # which are either itself or its super-clusters
        for e in clust:
            if commclust is None:
                commclust = invidx[e]
            else:
                commclust = commclust & invidx[e]

        commclust = set() if commclust is None else (commclust - {cid})
        # record all super-clusters
        superclusters = superclusters | commclust
    nosubclusters = list(set(range(len(clusters))) - superclusters)
    L = [l for cid, l in enumerate(clusters) if cid in nosubclusters]
    return L


def inverted_index(L):
    """Build inverted index of L."""
    invidx = defaultdict(set)
    for cid, (_, clust) in enumerate(L):
        for e in clust:
            invidx[e].add(cid)
    return invidx


def calc_committee_vectors(committees, freqmx=None):
    """Calculate committee vectors based on raw co-occurrence frequency matrix.
    For each committee, calculate the centroid vector of the committee (cluster),
    based on the frequency vectors of the elements in committee.
    Calculate the sum vector of frequency vectors of the rest elements.
    Based on these centroid vector and sum (rest) vector, calculate the ppmi vector
    of this committee.

    Parameters
    committee : iterable
        A list of committees (cluster list)
    freqmx : numpy.ndarray
        Raw co-occurrence frequency matrix.
        Notes : passing scipy.spmatrix would raise error
    """
    sum_all_vec = np.sum(freqmx, axis=0)  # row sum
    size = len(committees)
    commx = np.zeros((size, freqmx.shape[1]))  # -> committee matrix

    # this part is time consuming
    # for i, (maxv, clust) in progbar(enumerate(committees)):
    for i, clust in progbar(enumerate(committees)):
        if not isinstance(clust, list):
            clust = sorted(clust)
        # compute centroid vector
        centroid_vec = centroid_of_cluster(freqmx, cluster=clust)
        # sum the cluster vectors
        sum_clust_vec = sum_of_cluster(freqmx, cluster=clust)
        # collapse (sum) the rest vectors into one
        sum_rest_vec = sum_all_vec - sum_clust_vec
        # calculate (positive) pmi vector for cluster
        ppmi_vec = calc_meas_vec(centroid_vec, sum_rest_vec)
        commx[i] = ppmi_vec

    return commx


def cbc_step3_old(L, freqmx=None, theta=0.35):
    """Cluster by Committee Step 3.
    Let C be a list of committees, initially empty.
    For each cluster c of L in sorted order, compute the centroid of c.
    If c's similarity to the centroid of each committee previously added to C is below a threshold theta1, add c to C.
    This assures that the inter-cluster similarities are low.

    Parameters
    ----------
    L : list of list or numpy.array
    freqmx : numpy.ndarray
    theta : float
        This is the theta1 in CBC algorithm.
        If a new committee's similarities between each committee previously added is below theta, add this committee.

    Returns
    -------
    (C, commx) : (list, numpy.ndarray)
        ( clusters in L, committee vectors )

    """
    sum_all_vec = np.sum(freqmx, axis=0)
    size = len(L)
    highest_scores = []
    commx = np.zeros((size, freqmx.shape[1]))  # -> committee matrix

    # this part is time consuming
    for i, (maxv, clust) in progbar(enumerate(L)):
        highest_scores.append(maxv)
        # compute centroid vector
        centroid_vec = centroid_of_cluster(freqmx, cluster=clust)
        # sum the cluster vectors
        sum_clust_vec = sum_of_cluster(freqmx, cluster=clust)
        # collapse (sum) the rest vectors into one
        sum_rest_vec = sum_all_vec - sum_clust_vec
        # calculate (positive) pmi vector for cluster
        ppmi_vec = calc_meas_vec(centroid_vec, sum_rest_vec)
        commx[i] = ppmi_vec

    # now we have the committee matrix (frequency matrix based on raw frequency)
    # since L is sorted, so we can add the first cluster
    C = [0]  # for indices of committees to keep
    simmx = cosine_similarity(commx)  # cosine similarity matrix
    for i in progbar(range(1, simmx.shape[0])):
        below = True
        for j in C:
            if simmx[j, i] >= theta:  # found one's similarity >= theta
                below = False
                break
        if below:
            C.append(i)
    commx = commx[np.array(C)]
    C = [L[i] for i in C]

    return C, commx


def cbc_step3(elements, measmx=None, comms=None, commx=None, theta=0.25):
    """Cluster by Committee Step 5.
    Use association measure vector to represent elements.
    For each element e, if e's similarity to every committee in C is below threshold theta (theta2),
    add e to a list of residues R.

    Notes : when theta decreases, the number of residues increases.

    Parameters
    ----------
    elements : iterable (of str)
        A list of elements.
    measmx : numpy.ndarray
        Association measure matrix of elements
    comms : iterable (of lists of str)
        Committee sets of elements
    commx : numpy.ndarray
        Committee matrix
    theta : float
        Default 0.25

    Returns
    -------
    (R, comms) : tuple
        A list of residues and a list of committees.
    """
    R = []
    comms = deepcopy(comms)
    # calculate similarity between each element (represented by association measure vector)
    # and each committee
    logger.info("calculating similarity between elements and committees...")
    simmx = cosine_similarity(measmx, commx)
    maxidx = np.argmax(simmx, axis=1)  # max value index of each row

    for i in progbar(range(simmx.shape[0])):
        isresidue = True
        for j in range(simmx.shape[1]):
            if simmx[i, j] >= theta:  # if there is a similarity >= theta, it is not a residue
                isresidue = False
                break
        e = elements[i]
        if isresidue:
            R.append(e)
        else:
            if e not in comms[maxidx[i]]:
                comms[maxidx[i]].append(e)
    return R, comms


def cluster_similar_elements(e, distmx=None, k=100,
                             prune_method='distance', t='median',
                             score_metric='without_size', highest_score=False, score_func=None):
    """Cluster k similar elements of input element, based on distance matrix.

    Parameters
    ----------
    e : str
    distmx : :class:`~nephosem.TypeTokenMatrix`
    k : int
        Number of most similar elements to cluster.
    prune_method : str
        Method to prune the hierarchical clustering tree.
        ``distance`` :
            Prune the tree by drawing a horizontal line on height (distance) equaling to *t* (threshold).
        ``minsize`` :
            Prune the tree by drawing a horizontal line on a minimum height
            when there is one cluster containing more than (or equal to) *t* elements among clusters below this line.
    t : int or float or str
        The threshold to apply when pruning clustering tree.
    score_metric : str
        ``without_size`` : Scores a cluster by only its average similarity.
        ``with_size`` : Scores a cluster by the product of its size and its average similarity.
        ``with_size_sqrt`` : Scores a cluster by the product of the sqrt of its size and its average similarity.
    highest_score : bool
        If True, retrieve only the highest scoring cluster based on scoring metric.
        Else, retrieve all clusters by ``hierarchical_cluster`` function.
    score_func : function
        A function to perform on a cluster of elements.
        Basically two parameters are considered: average similarity of the cluster and the size of it.

    Returns
    -------
    clusters : iterable
        [ (score, cluster) ...]
        Notes : if ``highest`` is True, return only the cluster with highest score,
                else, return all clusters.
    """
    # get a sub matrix of k most similar elements
    sim_eles = distmx.most_similar(e, k=k)
    subdistmx = distmx.submatrix(row=sim_eles, col=sim_eles)
    cdsubdistmx = squareform(subdistmx.matrix)  # -> condensed distance matrix

    # hierarchically cluster the distance matrix and return all clusters of the last n levels
    clusters = hierarchical_cluster(cdsubdistmx, prune_method=prune_method, t=t)
    if len(clusters) == 0:
        return []

    # get highest score cluster among all clusters
    subsimmx = 1.0 - cdsubdistmx  # transform to a similarity arrays
    clusters_with_score = score_cluster(clusters, subsimmx, metric=score_metric, highest=highest_score, score_func=score_func)

    # item ids are all local ids in submatrix, so remap them to the large matrix
    glbclusts = []
    for s, c in clusters_with_score:
        glbclust = sub2glb(c, subdistmx.row_items, distmx.row_items)
        glbclust = [distmx.row_items[i] for i in glbclust]
        glbclusts.append((s, glbclust))

    return glbclusts


def sub2glb(cluster, subitems, glbitems):
    """Remap submatrix ids of a cluster to global matrix ids.

    Parameters
    ----------
    cluster : iterable
        A list of element indices as a cluster
    subitems : iterable
        Submatrix row items
    glbitems : iterable
        Global matrix row items

    Returns
    -------
    global cluster : a list of element indices
    """
    cluster_items = [subitems[i] for i in cluster]  # sub-indices -> items
    glbitem2id = {e: i for i, e in enumerate(glbitems)}
    glb_cluster = [glbitem2id[it] for it in cluster_items]  # -> global indices of items
    return glb_cluster


def hierarchical_cluster(distmx, prune_method='distance', t='median'):
    """Perform hierarchical clustering on distance matrix.

    Parameters
    ----------
    distmx : numpy.array
        flat numpy array
    prune_method : str
        Method to prune the hierarchical clustering tree.
        ``distance`` :
            Prune the tree by drawing a horizontal line on height (distance) equaling to *t* (threshold).
        ``minsize`` :
            Prune the tree by drawing a horizontal line on a minimum height
            when there is one cluster containing more than (or equal to) *t* elements among clusters below this line.
    t : int or float
        The threshold to apply when pruning clustering tree.

    Returns
    -------
    clusters : dict
        { cluster id -> list of element ids }

    """
    Z = sch.linkage(distmx, method='average')
    clusts = flat_cluster(Z, t=t, criterion=prune_method)
    clusters = defaultdict(list)  # cluster index -> a list of element indices
    for i, c in enumerate(clusts):
        clusters[c].append(i)
    # clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    return clusters


def flat_cluster(Z, t='median', criterion='distance', depth=2):
    """Get clusters which contains element ids from Z.
    Prune the hierarchical clustering tree by criterion and threshold.

    Parameters
    ----------
    Z : numpy.ndarray
        The hierarchical clustering encoded as a linkage matrix.
    t : float or int or str
        The threshold to apply when forming flat clusters.
    criterion : str, optional
        The criterion to use in forming flat clusters. This can be any of the following values:
            ``minsize`` :
                Forms flat clusters so that
                when draws a horizontal line of a threshold distance to the dendrogram,
                there is one cluster containing at least t elements.

            ``distance`` :
                Forms flat clusters so that the original
                observations in each flat cluster have no greater a cophenetic distance than t.

            ``maxclust`` :
                Finds a minimum threshold r so that
                the cophenetic distance between any two original observations in the same flat cluster is no more than
                r and no more than t flat clusters are formed.
    depth : int, optional
        The maximum depth to perform the inconsistency calculation. It has no meaning for the other criteria.
        Default is 2.
        ``Notes`` : currently not used

    Returns
    -------
    flat cluster : numpy.ndarray
        An array of length n. T[i] is the flat cluster number to which original observation i belongs.
    """
    # hierarchical clustering based on distmx
    clusters = []
    if criterion == 'minsize':
        # Bottom-up the cluster tree, when there is a cluster who has more than k elements,
        # stop and draw a horizontal line to cut the higher merges.
        threshold = 1.0
        for idx1, idx2, dist, count in Z:
            if count >= t:
                threshold = dist
                break
        clusters = sch.fcluster(Z, threshold, criterion='distance')
    elif criterion == 'distance':
        if isinstance(t, str):
            mergedist = np.array([d for _1, _2, d, _4 in Z])  # a list of merge distances
            if t == 'mean':
                threshold = np.mean(mergedist)
            elif t == 'median':
                threshold = np.median(mergedist)
            elif '%' in t:  # -> '25%', '75%'
                threshold = int(t[:t.index('%')])
                threshold = np.percentile(mergedist, threshold)
            else:
                raise NotImplementedError("Not implemented this threshold type!")
        else:
            threshold = t
        clusters = sch.fcluster(Z, threshold, criterion='distance')
    else:
        pass
    '''
    clusters = dict()
    for i in range(1, len(Z)+1):
        fcids = sch.fcluster(Z, i, criterion='maxclust')
        level_clusters = reverse(fcids, i)
        clusters.update(level_clusters)
    '''
    return clusters


def score_cluster(clusters, simmx, metric='with_size', highest=False, score_func=None):
    """Return the cluster (representing by all element ids) which has the maximum score.

    Parameters
    ----------
    clusters : dict
        { cluster id -> list of element ids }
    simmx : numpy.ndarray
    metric : str
        ``without_size`` : Scores a cluster by only its average similarity.
        ``with_size`` : Scores a cluster by the product of its size and its average similarity.
        ``with_size_sqrt`` : Scores a cluster by the product of the sqrt of its size and its average similarity.
    highest : bool
        If True, only return the cluster with highest score.
        Else, return all clusters with their scores.
    score_func : function
        A function to calculate the score.
        Basically take two parameters: the average similarity of the cluster and the size of it.
        ``Example`` :
            def func(size, avgsim):
                return math.sqrt(size) * avgsim

            The effect of this function is the same as metric='with_size_sqrt'.

    """
    cluster_scores = {}
    clustsims = cluster_avgsim(clusters, simmx)
    maxc, maxs = -1, 0.0
    for cid, ids in clusters.items():
        if len(ids) == 1:  # skip single element
            cluster_scores[cid] = -1  # give -1 score to singleton cluster
            continue

        avgsim = clustsims[cid]
        if metric == 'with_size_sqrt':
            score = math.sqrt(len(ids)) * avgsim
        elif metric == 'with_size':
            score = len(ids) * avgsim
        elif metric == 'without_size':
            score = avgsim
        else:
            raise NotImplementedError("Not implemented: {}".format(metric))

        cluster_scores[cid] = score
        if score > maxs:
            maxs = score
            maxc = cid

    if highest:
        return [(maxs, clusters[maxc])]
    else:
        return [(cluster_scores[cid], clust) for cid, clust in clusters.items()]


def cluster_avgsim(clusters, simmx):
    """Compute average similarities for all clusters based on similarity matrix.

    Parameters
    ----------
    clusters : dict
        A python dict whose key is a cluster index and value is element indices of that cluster.
    simmx : numpy.ndarray
        Condensed similarity array.
        See `scipy.spatial.distance.squareform`.
        simmx ->
        array([  1., 5., 15.5241747, 4.47213595, 14.56021978,  12. ])
        squareform(simmx) ->
        array([[  0.        ,   1.        ,   5.        ,  15.5241747 ],
               [  1.        ,   0.        ,   4.47213595,  14.56021978],
               [  5.        ,   4.47213595,   0.        ,  12.        ],
               [ 15.5241747 ,  14.56021978,  12.        ,   0.        ]])

    Returns
    -------

    """
    avgsim = {}
    for c, idx in clusters.items():
        if len(idx) == 1:
            avgsim[c] = 1.0
        else:
            avgsim[c] = average_similarity(idx, simmx)
    return avgsim


def average_similarity(idx, simmx):
    """Compute average similarity of a cluster (a list of element indices).

    Parameters
    ----------
    idx : iterable or numpy.ndarray
        A list of element indices.
    simmx : numpy.ndarray
        Condensed array of similarity matrix.
    """
    n = int(math.sqrt(len(simmx) * 2)) + 1
    size = len(idx)
    csize = int(size * (size - 1) / 2)
    sum = 0.0
    for i in range(size):
        for j in range(i+1, size):
            id1, id2 = idx[i], idx[j]
            sum += simmx[square_to_condensed(id1, id2, n)]
    return sum / csize


def square_to_condensed(i, j, n):
    """Transform square indices i, j to condensed index."""
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)/2 + i - 1 - j)


def calc_meas_vec(target_vec, rest_vec, meas='ppmi'):
    """Calculate association measure values for two vectors
    Returns a vector of association measure (ppmi) which has same dimension of the two input vectors

    Parameters
    ----------
    target_vec : numpy.ndarray
        1-D array
    rest_vec : numpy.ndarray
        1-D array
    """
    target_sum = target_vec.sum()
    rest_sum = rest_vec.sum()
    tot_sum = target_sum + rest_sum

    row_sum = target_sum / tot_sum
    col_sum_vec = np.sum([target_vec, rest_vec], axis=0) / tot_sum
    norm_target_vec = target_vec / tot_sum

    size = target_vec.shape[0]
    meas_vec = np.zeros(target_vec.shape[0])
    for i in range(size):
        if norm_target_vec[i] == 0:
            x = 0.0
        else:
            x = norm_target_vec[i] / (row_sum * col_sum_vec[i])
        pmi = math.log(x if x != 0.0 else 1.0)
        if meas == 'ppmi':
            meas_vec[i] = 0.0 if pmi < 0.0 else pmi

    return meas_vec


def reverse(fcids, lid):
    """
    # lid : level id
    # fcids: id -> cid
    Parameters
    ----------
    fcids : iterable
        flat cluster index -> its element indices
    lid : int
        level index (in hierarchical clustering)

    Returns
    -------

    """

    clusters = defaultdict(set)
    for i, cid in enumerate(fcids):
        clusters[(lid, cid)].add(i)
    return clusters


def save_committees_json(tmpdir, comms):
    C_fname = os.path.join(tmpdir, 'Cs')
    with codecs.open(C_fname, 'w') as fout:
        json.dump(comms, fout, ensure_ascii=False, indent=4)


def save_residues_json(tmpdir, residues):
    R_fname = os.path.join(tmpdir, 'Rs')
    with codecs.open(R_fname, 'w', encoding='latin1') as fout:
        json.dump(residues, fout, ensure_ascii=False, indent=4)
