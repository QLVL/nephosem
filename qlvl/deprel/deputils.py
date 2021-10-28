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

import os
import re
import codecs
from itertools import product

from copy import deepcopy
from collections import defaultdict
# from pyspark import SparkContext
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

# from .basic import Path, Graph


def split_large_file(filename, encoding='utf-8'):
    """Split large corpus file into samller ones for multicore processing"""
    i = 0
    basedir, ext = os.path.splitext(filename)
    os.makedirs(basedir, exist_ok=True)
    basename = os.path.basename(filename)
    with codecs.open(filename, 'r', encoding=encoding) as fin:
        num_sents = 0
        chunk = []
        for line in fin:
            chunk.append(line.strip())
            if line.startswith('</s'):  # end of a sentence
                num_sents += 1
                if num_sents % 100000 == 0:
                    split_fname = '{}/{}.{}'.format(basedir, basename, i)
                    with codecs.open(split_fname, 'w', encoding=encoding) as fout:
                        fout.write('\n'.join(chunk))
                    chunk = []
                    print(i, end=' ')
                    i += 1


'''
def match_v1(g, p):
    validSet = set()
    for n in g.nodes_iter():
        result = re.match(p.nodes[0], n)
        if not result:
            continue
        # start from this node and check if graph matches the path
        # Breadth search
        validSeq = [n]
        matchPath_v1(n, g, p, 0, validSeq, validSet)
    return validSet

def matchPath_v1(cur, graph, path, step, validSeq, validSet):
    if step == path.len:
        for v in validSeq:
            validSet.add(v)
        return True
    for neighbor in nx.all_neighbors(graph, cur):
        rel = graph[cur][neighbor]['rel']
        if not re.match(path.edges[step], rel) or not re.match(path.nodes[step+1], neighbor):
            continue
        validSeq.append(neighbor)
        matchPath_v1(neighbor, graph, path, step+1, validSeq, validSet)
    if len(validSeq) != path.len + 1:
        validSeq.remove(cur)
'''




def read_nodes(in_nodes, encoding='utf-8'):
    """Read nodes from file.
    i.e. :
        a/DT    4
        an/DT   21
        ...
    """
    node2id, id2node = dict(), dict()
    with codecs.open(in_nodes, encoding=encoding) as fin:
        for line in fin:
            s, idx = line.strip().split("\t")
            node2id[s] = idx
            id2node[idx] = s

    return node2id, id2node


def process_sentence(sLine):
    """Process a sentence line.
    line : 'sid:edge1;edge2;edge3...'

    Parameters
    ----------
    sLine

    Returns
    -------

    """
    res = []
    sid, edges = sLine.split(':')  # split sentence id and edges
    edges = str(edges).strip().split(';')  # split edges
    g = Graph(edges, id2node)  # id2node is global
    pid = 1  # path index
    for p in paths:
        valid_matches = match_graph(g.graph, p)
        for m in valid_matches:
            # [pid, sid, node1, node2, ...]
            tmp = [str(pid), str(sid)] + [str(node2id[x] for x in m)]  # map(lambda x: str(node2id[x]), m)
            res.append(tmp)
        pid += 1
    return res


def cartesian_product(mapping):
    """
    {1: (1, 2), 2: (3, 4)} -> [{1:1, 2:3}, {1:2, 2:3}, {1:1, 2:4}, {1:2, 2:4}]
    Should perform:
    {1: (1, 2), 2: (1, 2)} ->
    [{1: 1, 2: 2}] (if the target is not in (1, 2))
    [{1: 1, 2: 2}, {1: 2, 2: 1}]
    """
    combinations = []
    for fn, snodes in mapping.items():
        pairs = [(fn, sn) for sn in snodes]
        combinations.append(pairs)
    combinations = product(*combinations)  # possible mapping pairs of this level
    return [{fn:sn for fn, sn in match} for match in combinations]


def judgeOut(sLine):
    res_dict = defaultdict(lambda: defaultdict(set))
    sid, edges = sLine.split(':')  # sid -> sentence id
    edges = edges.strip().split(';')  # edges -> a list of 'from_node,rel,to_node'
    g = Graph(edges, id2node)
    pid = 1
    for p in paths:
        validSet = match_graph(g.graph, p)
        for node in validSet:
            res_dict[node][pid].add(sid)
        pid += 1
    res = []
    for node, v in res_dict.items():
        for pid, sid in v.iteritems():
            res.append((node, pid, sid))
    return res


def judgeIn(sLine):
    res = []
    sid, edges = sLine.split(':')
    edges = str(edges).strip().split(';')
    g = Graph(edges, id2node)
    #
    for pid, p in enumerate(paths):
        validMatches = match_graph(g.graph, p)
        # when there is only one match in the sentence
        # set the integer sid to sid.0
        # when we see '.0' after one sid
        # we know there is only one match in this sentence for path
        if len(validMatches) == 1:
            for j, node in enumerate(validMatches[0]):
                f_pid = str(float(pid) + 1.0 + (j + 1) * 0.1)
                t = (str(node), (f_pid, str(float(sid))))
                res.append(t)
        else:
            for i, m in enumerate(validMatches):
                for j, node in enumerate(m):
                    f_pid = str(float(pid) + 1.0 + (j + 1) * 0.1)
                    f_sid = str(float(sid) + (i+1)*0.1)
                    t = (str(node), (f_pid, f_sid))
                    res.append(t)
    return res


def judgeIn_v1(sLine):
    res_dict = defaultdict(lambda: defaultdict(int))
    sid, edges = sLine.split(':')
    edges = str(edges).strip().split(';')
    g = Graph(edges, id2node)
    pid = 1
    for p in paths:
        validSet = match_v1(g.graph, p)
        for node in validSet:
            res_dict[node][pid] = sid
        pid += 1
    res = []
    for node, v in res_dict.iteritems():
        for pid, sid in v.iteritems():
            res.append((node, (pid, sid)))
    return res


def group(nodemap):
    node, value = nodemap
    d = defaultdict(set)
    for pid, sid in value:
        d[pid].add(sid)
    return node, d


def outMap(item):
    node, d = item
    mapped = map(lambda x: '{}:[{}]'.format(x[0], ','.join(sorted(x[1]))), sorted(d.iteritems()))
    return '{}->{}'.format(node, ';'.join(mapped))

