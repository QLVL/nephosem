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


import os
import re
import codecs
import logging
from collections import deque
from copy import deepcopy
from collections import defaultdict

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

__all__ = ['parse_pattern', 'get_root', 'get_depth', 'tree_match',
           'draw_tree', 'draw_labels', 'draw_match']

logger = logging.getLogger(__name__)


def find_arrow(s, start, larrow='<-', rarrow='->'):
    """Find the next index of an arrow character"""
    arrow_len = len(larrow)
    for i in range(start, len(s)):
        if s[i:i+arrow_len] == larrow or s[i:i+arrow_len] == rarrow:
        # if s[i] in '<>':
            # if there is a '[' character before this arrow
            # find the next arrow after ']'
            if '[' in s[start:i] and ']' not in s[start:i]:
                continue
            else:
                return i
    # reach the end of string, there is not a next arrow
    return len(s)


def parse_sub(prev_nid, subfeature, arrow, node_dict, edge_dict,
              larrow='<-', rarrow='->'):
    if len(subfeature.strip()) <= 0:
        return

    arrow_len = len(larrow)
    # the sub-feature would be either a list of nodes or a sub-tree
    if subfeature[0] == '[' and subfeature[-1] == ']':
        # list of nodes: [(acomp) (\w+)/(JJ) , (nsubj) (\w+)/(N)\w*]
        next_nodes = subfeature[1:-1].split(',')
        next_nodes = [n.strip() for n in next_nodes]
        for node in next_nodes:
            node = node.strip()
            parse_sub(prev_nid, node, arrow, node_dict, edge_dict)
    else:
        # sub-tree: (nsubj) (\w+)/(V)\w* >[(prep) (\w+)/(IN) >[(pobj) (\w+)/(N)\w*]]
        # 1. fetch dependency relation and the next node
        arrow_idx = find_arrow(subfeature, 0)  # the next arrow index
        next_node = subfeature[:arrow_idx].strip()
        rel, next_node = next_node.split(' ', 1)
        rel = rel.strip(); next_node = next_node.strip()
        next_nid = len(node_dict) + 1
        node_dict[next_nid] = next_node
        # 2. add an edge
        if arrow == larrow:
            # edge_dict[f'{next_nid}->{prev_nid}'] = rel
            edge_dict[(next_nid, prev_nid)] = rel
        else:
            # edge_dict[f'{prev_nid}->{next_nid}'] = rel
            edge_dict[(prev_nid, next_nid)] = rel

        if arrow_idx == len(subfeature):
            # end of feature
            return
        else:
            next_arrow = subfeature[arrow_idx:arrow_idx+arrow_len]
            parse_sub(next_nid, subfeature[arrow_idx+arrow_len:], next_arrow, node_dict, edge_dict)


def parse_pattern(target, feature, larrow='<-', rarrow='->'):
    """Parse a string of feature regex to a node dict and an edge dict.
    e.g target = '(\w+)/(N)\w*', feature = '<(nsubj) \w+/(V)\w* >[(acomp) (\w+)/(JJ)]'
    ==>

    """
    target = target.strip()
    feature = feature.strip()

    # read the first character of feature regex, which should be an arrow
    node_dict = {}
    edge_dict = {}
    # add the first target node
    prev_node = target
    prev_nid = 1
    node_dict[prev_nid] = prev_node  # 1-based

    arrow_len = len(larrow)
    arrow = feature[:arrow_len]  # first character is an arrow '<' or '>'
    subfeature = feature[arrow_len:].strip()  # rest feature
    parse_sub(prev_nid, subfeature, arrow, node_dict, edge_dict)

    return node_dict, edge_dict


def get_root(gx):
    """Get the root of a tree"""
    v = 1
    while True:
        ps = gx.predecessors(v)  # there should always be one predecessor
        try:
            tmp = next(ps)
        except:
            break
        v = tmp
    return v


def get_depth(gx):
    """Get the depth of a tree"""
    maxdepth = 1
    for v in gx.nodes:
        if gx.out_degree(v) == 0:
            # a leaf
            depth = get_depth_of_node(gx, v)
            if depth > maxdepth:
                maxdepth = depth
    return maxdepth


def get_depth_of_node(gx, v):
    """Get the depth of a node"""
    cur = v
    depth = 1
    while True:
        preds = gx.predecessors(cur)  # there should only be one predecessor
        try:
            pred = next(preds)
        except:
            break
        cur = pred
        depth += 1
    return depth


def tree_match(sentence, macro):
    """Match the sentence with the subgraph pattern.
    Abstract of the matching:
        * find root node of the pattern
        * iterate over each sentence node that matches the pattern root
        * recursively match the sentence with the pattern from each matched node

    Parameters
    ----------
    sentence : :class:`~qlvl.core.graph.SentenceGraph`
    macro : :class:`~qlvl.core.graph.MacroGraph`
    """
    # check whether the sentence (dependency tree) is a valid tree
    try:
        assert sentence.istree
    except Exception as err:
        logger.error(err, str(sentence.fid))
        return False

    num_curr_matches = len(macro.matched_nodes)
    # find feature root
    froot = get_root(macro.graph)

    # iterate over each sentence node that matches the root of the feature
    for v, vitem in sentence.nodes:
        if not macro.match_node(vitem, idx=froot):
            continue
        mapping = {froot: v}  # mapping of the first level
        # recursively match the sentence with the feature from this node
        subtree_match(sentence=sentence, macro=macro, lmatches=deque([[mapping]]))

    return not num_curr_matches == len(macro.matched_nodes)


def subtree_match(sentence=None, macro=None, lmatches=None):
    """Match the sentence with a subtree of the feature by a level-match algorithm.
    Abstract:
        * Get a match from the `lmatches` queue
        * Match the next level of the sentence based on the matched mapping of current level
        * Append the mapping of next level to the current match
        * Recursively match the next level

    Parameters
    ----------
    sentence : :class:`~qlvl.core.graph.SentenceGraph`
    macro : :class:`~qlvl.core.graph.MacroGraph`
    lmatches : queue (collections.deque)
        Contains a list of possible matches.
        Each match is a (finally the length is `pattern.depth`) lists of levels of the pattern.
        Element example: pattern node idx -> sentence node idx.
    """
    # if find a match
    if len(lmatches) <= 0:
        return
    # the current depth of matching
    match_depth = len(lmatches[0])
    if match_depth == macro.depth:  # the last level
        add_match(sentence, lmatches, macro)
        return

    size = len(lmatches)  # number of possible matches
    for _ in range(size):
        lmatch = lmatches.popleft()  # get each possible match (of all levels)
        currmap = lmatch[-1]  # node index mapping of current level
        # match the next level of the sentence based on the matched 'mapping' of current level
        matched, matched_succs = match_level(sentence, macro, currmap)
        if not matched:
            continue
        # append the map of next level to current matches
        # for each possible map of next level, create a new match of previous levels and append the new map
        for sucmap in matched_succs:
            newmap = deepcopy(lmatch)
            newmap.append(sucmap)
            lmatches.append(newmap)

    # recursively match the next level
    subtree_match(sentence=sentence, macro=macro, lmatches=lmatches)


def add_match(sentence, lmatches, macro):
    """When found a new match, add this match to macro's matched_nodes and matched_edges"""
    for lmatch in lmatches:
        # merge dicts of levels
        nodemapping = dict()
        for m in lmatch:
            nodemapping.update(m)
        # remap feature node to sentence node label (type)
        matched_nodes = {fn: sentence.nodes[sn] for fn, sn in nodemapping.items()}
        # if the target filter is given, check whether the matched target node appear in the target filter
        if macro.target_filter is not None:
            target_node_dict = matched_nodes[macro.target_idx]
            # example of target_node_dict: {'FORM': 'Het', 'LEMMA': 'het', 'POS': 'det'}
            vals = [target_node_dict[attr] for attr, _idx in macro.target_node_attrs.items() if attr not in ['FID', 'LID', 'fid', 'lid']]
            target_type = macro.connector.join(vals)
            # if the matched target type does not appear in the given target filter
            # do not add this match to the macro (for speeding up processing and save memory space)
            if target_type not in macro.target_filter:
                continue
        if sentence.mode == 'token':
            for nid, attrs in matched_nodes.items():
                attrs['FID'] = sentence.fid
                attrs['fid'] = sentence.fid
        # remap feature edge (tuple of node idx) to sentence edge label (dependency relation)
        matched_edges = {(head, tail): sentence.edges[(nodemapping[head], nodemapping[tail])]
                         for head, tail in macro.edges}
        # append matched nodes and edges to the feature
        macro.add_match(matched_nodes, matched_edges)


def match_level(sentence, pattern, prevmap):
    """Match the current level based on the index mapping (pattern index -> sentence index) of previous level.
    Abstract:
        * Match the successors of a sentence node with the successors of the corresponding pattern node
        *

    Parameters
    ----------
    sentence : :class:`~qlvl.core.graph.SentenceGraph`
    pattern : :class:`~qlvl.core.graph.MacroGraph`
    prevmap : dict
        Index mapping from sentence node to pattern node (of previous level).
        e.g. pattern node idx -> sentence node idx

    Returns
    -------
    A list of dicts : [{pattern node idx -> sentence node idx, ...}, ...]
    """
    currmaps = deque([dict()])  # node mapping of current level
    for fn, sn in prevmap.items():
        # if reach a leaf of the feature
        if pattern.out_degree(fn) == 0:
            # then there is no need to matched its next level (no successors)
            continue
        # matched successors of 'sn' (sentence node) with successors of 'fn' (feature node)
        succmatched, succmaps = match_successors(sentence, sn, pattern, fn)
        if succmatched:
            # there might be several matches of (fn, sn)
            size = len(currmaps)
            for _ in range(size):
                prevone = currmaps.popleft()
                for onemap in succmaps:
                    newmap = deepcopy(prevone)
                    newmap.update(onemap)
                    currmaps.append(newmap)
        else:  # there is no match in the next level
            return False, None

    return True, list(currmaps)


def match_successors(sentence, scur, pattern, fcur):
    """Match sentence successors with pattern successors based on current sentence node and pattern node.
    All pattern successor nodes have to be matched.

    Parameters
    ----------
    sentence : :class:`~qlvl.core.graph.SentenceGraph`
    scur : int
        Current node index of sentence
    pattern : :class:`~qlvl.core.graph.MacroGraph`
    fcur : int
        Current node index of pattern

    Returns
    -------
    A list of dicts : [{pattern node idx -> sentence node idx, ...}, ...]
    """
    mapping = deque([dict()])
    # return if reach a leaf
    if pattern.out_degree(fcur) == 0:
        return True, mapping

    # we have a list of successors of current sentence node and a list of successors of current pattern node
    # we need to match these sentence nodes with pattern nodes
    for fsuc in pattern.successors(fcur):
        size = len(mapping)
        i = 0
        while i < size:
            onemap = mapping.popleft()
            msuccs = []
            for ssuc in sentence.successors(scur):
                if ssuc in onemap:
                    continue
                vitem = sentence.nodes[ssuc]
                eitem = sentence.edges[(scur, ssuc)]
                vmatched = pattern.match_node(vitem, idx=fsuc)
                ematched = pattern.match_edge(eitem, idx=(fcur, fsuc))
                # if not matched, the returned vmatched or ematched should be None
                if vmatched and ematched:
                    msuccs.append(ssuc)
            # TODO: check for symmetry
            if len(msuccs) > 0:
                for ssuc in msuccs:
                    newmap = deepcopy(onemap)
                    newmap[ssuc] = fsuc
                    mapping.append(newmap)
            i += 1

    mapping = [{v: k for k, v in m.items()} for m in mapping]
    return True if len(mapping) > 0 else False, mapping


def match_graph(graph, path):
    """Match a graph with path.

    Parameters
    ----------
    graph : networkx.Graph
    path : :class:`Path`

    Returns
    -------
    valid matches : iterable
    """
    valid_matches = []
    # iterate over all nodes of graph
    for nd in graph.nodes():
        # find the head node
        if not path.match_node(nd):
            continue
        # start from this node and check if graph matches the path
        valid_seq = [nd]
        # Breadth search
        match_path(nd, graph=graph, path=path, step=0, valid_seq=valid_seq, valid_matches=valid_matches)
    return valid_matches


def match_path(cur, graph=None, path=None, step=0, valid_seq=None, valid_matches=None):
    """Match path in a graph from a start node.

    Parameters
    ----------
    cur : str
        Current node
    graph : networkx.Graph
    path : :class:`Path`
    step : int
        Current step number
    valid_seq : iterable
        Valid sequence
    valid_matches : iterable
        Valid matches
    """
    if step == path.len:
        valid_matches.append(valid_seq)
        return

    # iterate all neighbors of current node
    for neighbor in nx.all_neighbors(graph, cur):
        rel = graph[cur][neighbor]['rel']  # -> relation label between current node and neighbor
        # check if edge and node match path
        if (not path.match_edge(graph[cur][neighbor], idx=step)
            or not path.match_node(graph[neighbor], idx=step+1)):
        # if not re.match(path.edges[step], rel) or not re.match(path.nodes[step+1], neighbor):
            continue
        # as match an edge (and 'to' node), append neighbor to valid sequence
        valid_seq.append(neighbor)
        # recursively match graph and path
        match_path(neighbor, graph=graph, path=path, step=step + 1, valid_seq=valid_seq, valid_matches=valid_matches)

    if len(valid_seq) != path.len + 1:
        valid_seq.remove(cur)


# draw functions
def draw_tree(gx, v_label=None, e_label=None, figsize=(5.0, 5.0)):
    """Draw a tree layout of a graph
    Parameters
    ----------
    gx : networkx.DiGraph
    v_label : str
    e_label : str
    figsize : tuple of two values
    """
    plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
    pos = graphviz_layout(gx, prog='dot')

    nx.draw_networkx_nodes(gx, pos, node_color='r', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(gx, pos, width=7, alpha=0.5, edge_color='b')

    if isinstance(v_label, str):
        v_labels = {v: gx.nodes[v].get(v_label, '') for v in gx.nodes()}
    elif isinstance(v_label, list):
        v_labels = {v: '/'.join([gx.nodes[v].get(lb, '') for lb in v_label])
                    for v in gx.nodes}
    else:
        raise ValueError('v_label should be either string or list!')
    nx.draw_networkx_labels(gx, pos, v_labels, font_size=13)
    e_labels = {e: gx.edges[e].get(e_label, '') for e in gx.edges()}
    nx.draw_networkx_edge_labels(gx, pos, e_labels, font_size=13)

    plt.axis('off')
    plt.show()


def draw_match(feature, idx=0, figsize=(5.0, 5.0)):
    plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
    gx = feature.graph
    # pos = nx.spring_layout(gx) # positions for all nodes
    pos = graphviz_layout(gx, prog='dot')

    # nx.draw_networkx_nodes(gx, pos, node_color='r', node_size=500, alpha=0.8)
    colors = ['g'] + ['r'] * (gx.number_of_nodes() - 1)
    nx.draw_networkx_nodes(gx, pos, node_color=colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(gx, pos, width=7, alpha=0.5, edge_color='b')

    v_labels = feature.matched_nodes[idx]
    e_labels = feature.matched_edges[idx]
    nx.draw_networkx_labels(gx, pos, v_labels, font_size=13)
    nx.draw_networkx_edge_labels(gx, pos, e_labels, font_size=13)

    plt.axis('off')
    plt.show()


def draw_labels(gx, v_labels=None, e_labels=None):
    # pos = nx.spring_layout(gx) # positions for all nodes
    pos = graphviz_layout(gx, prog='dot')

    nx.draw_networkx_nodes(gx, pos, node_color='r', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(gx, pos, width=7, alpha=0.5, edge_color='b')

    if isinstance(v_labels, str):
        lbl = v_labels
        v_labels = {}
        for v in gx.nodes():
            v_labels[v] = gx.nodes[v].get(lbl, '')
    if isinstance(e_labels, str):
        lbl = e_labels
        e_labels = {}
        for e in gx.edges():
            e_labels[e] = gx.edges[e].get(lbl, '')

    nx.draw_networkx_labels(gx, pos, v_labels, font_size=11)
    nx.draw_networkx_edge_labels(gx, pos, e_labels, font_size=13)

    plt.axis('off')
