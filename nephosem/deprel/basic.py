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


import logging
import re
from collections import deque
from copy import deepcopy

import networkx as nx

from qlvl.specutils.deputils import match_graph, draw_tree, draw_match

logger = logging.getLogger(__name__)


class Sentence(object):
    def __init__(self, s):
        self.text = s
        self.content = []
        self.parse()

    def parse(self, s=None):
        """parse sentence text (raw string from corpus file)"""
        if s is None:
            s = self.text
        for line in s.split("\n"):
            if line.startswith('<'):  # skip xml tag lines
                continue
            self.content.append(line)

    def get_content(self):
        return self.content


class Graph(object):
    def __init__(self, sentence=None, id2node=None):
        """Construct a graph by a sentence and id2node.

        Parameters
        ----------
        sentence : iterable
        id2node : dict
        """
        self.graph = nx.Graph()
        self.node2id = {}
        if sentence:
            # self.build_graph(sentence, id2node)
            self.build_graph_raw(sentence)

    @property
    def nodes(self):
        return self.graph.nodes(data=True)

    @property
    def edges(self):
        return self.graph.edges(data=True)

    def add_node(self, v_id, v_label=None):
        """Add a node with id and label (optional) to graph."""
        if v_label:
            self.graph.add_node(v_id, label=v_label)
        else:
            self.graph.add_node(v_id)

    def add_edge(self, e_from_node, e_to_node, e_label):
        """Add an edge to graph."""
        self.graph.add_edge(e_from_node, e_to_node, rel=e_label)

    def build_graph_raw(self, sentence):
        """Build a graph from raw text (of a sentence)

        Parameters
        ----------
        sentence : iterable
            A list of strings
        """
        id2node = {0: 'root'}
        edges = []
        for line in sentence:
            eles = line.strip().split('\t')
            if len(eles) != 6:  # TODO: parse corpora of different formats
                continue
            word, pos, type_, idx, to_idx, rel = eles
            from_idx, to_idx = int(idx), int(to_idx)
            from_s = '{}/{}'.format(type_, pos)  # TODO: use CorpusFormatter
            id2node[from_idx] = from_s
            # we only have the 'to' idx, so might not have the 'to' node string
            # so we add all node strings first and record edges
            self.graph.add_node(from_s)  # add 'from' node first
            edges.append((from_idx, to_idx, rel))

        for e in edges:
            from_idx, to_idx, rel = e
            self.add_edge(id2node[from_idx], id2node[to_idx], rel)

    def build_graph(self, sentence=None, id2node=None):
        """Build a graph

        Parameters
        ----------
        sentence : iterable
            A list of dependency relations.
        id2node : dict
            Node id to node string mapping.
        """
        for line in sentence:
            from_node, rel, to_node = line.strip().split(',')
            from_s = id2node[from_node]
            to_s = id2node[to_node]
            self.graph.add_node(from_s)  # add 'from' node first
            if not self.graph.has_node(to_s):
                self.add_node(to_s)
            self.add_edge(from_s, to_s, rel)

    def match(self, path):
        """Match a graph with path.

        Parameters
        ----------
        path : :class:`PathTemplate`

        Returns
        -------
        valid matches : iterable
        """
        valid_matches = match_graph(self.graph, path)
        return valid_matches

    def __repr__(self):
        output = ""
        for v in self.graph.nodes():
            output += '\nv: {id}'.format(id=v)
            # output += '\nv {id} {label}'.format(id=v, label=l['label'])
        for e1, e2, rel in self.graph.edges(data='rel'):
            output += '\ne: {e1} {e2} {rel}'.format(e1=e1, e2=e2, rel=rel)
        output += '\n'
        return output


class PathTemplate(object):
    """Class representing a path template"""
    def __init__(self, nodes, edges):
        self.nodes = deepcopy(nodes)  # a list of str ->
        self.edges = deepcopy(edges)  # a list of str ->
        # self.nodes = map(lambda x: re.compile(x), nodes)
        # self.edges = map(lambda x: re.compile(x), edges)

    @property
    def len(self):
        return len(self.edges)

    def match_node(self, item, index=0):
        m = re.compile(self.nodes[index]).match(item)
        return m

    def match_edge(self, rel, index=0, u=0, v=0):
        m = re.compile(self.edges[index]).match(rel)
        return m

    def __str__(self):
        eles = [self.nodes[0]]
        for i in range(self.len):
            eles.append(self.edges[i])
            eles.append(self.nodes[i+1])
        return ':'.join(eles)


class Path(object):
    """Class storing path matches found in corpus"""
    def __init__(self, template, matches=None):
        self.template = template
        self.matches = matches if matches else []

    @property
    def len(self):
        """size of template
        i.e. '*:NN:amod:VB:*' has a size of one
        """
        return self.template.len

    @property
    def size(self):
        """number of matches"""
        return len(self.matches)

    def add_path(self, match):
        """Add a match

        Parameters
        ----------
        match : iterable
            A list of str
        """
        self.matches.append(match)

    def save(self, filename, encoding='utf-8'):
        pass

    @classmethod
    def load(cls, filename):
        pass


class DiGraph(object):
    def __init__(self):
        """Construct a graph by a sentence and id2node."""
        self.graph = nx.DiGraph()

    @property
    def nodes(self):
        return self.graph.nodes(data=True)

    @property
    def edges(self):
        # return self.graph.edges(data=True)
        return self.graph.edges

    @property
    def istree(self):
        # return nx.is_tree(self.graph)
        return nx.is_tree(self.graph) if not nx.is_empty(self.graph) else False

    def out_degree(self, v):
        return self.graph.out_degree(v)

    def in_degree(self, v):
        return self.graph.in_degree(v)

    def successors(self, v):
        return self.graph.successors(v)

    def predcessors(self, v):
        return self.graph.predecessors(v)

    def add_node(self, v_id, v_label):
        """Add a node with id and label (optional) to graph."""
        self.graph.add_node(v_id, label=v_label)

    def add_edge(self, e_id, from_v, to_v, e_label):
        """Add an edge to graph."""
        self.graph.add_edge(from_v, to_v, id=e_id, rel=e_label)

    def __repr__(self):
        output = ""
        for v in self.graph.nodes():
            output += '\nv: {id}'.format(id=v)
            # output += '\nv {id} {label}'.format(id=v, label=l['label'])
        for e1, e2, rel in self.graph.edges(data='rel'):
            output += '\ne: {e1} {e2} {rel}'.format(e1=e1, e2=e2, rel=rel)
        output += '\n'
        return output


class SentenceGraph(DiGraph):
    def __init__(self, nodes=None, edges=None, sentence=None):
        """Construct a graph by a sentence and id2node.

        Parameters
        ----------
        nodes : dict
        edges : dict
        sentence : iterable
            A list of strings (each string is a line in the corpus file)
        """
        super(SentenceGraph, self).__init__()
        if nodes and edges:
            self.generate_graph(nodes, edges)
        elif sentence:
            self.build_graph(sentence)
        else:
            raise ValueError("Please provide a sentence!")

    def generate_graph(self, nodes, edges):
        for idx, v in nodes.items():
            self.add_node(idx, v)
        for idx, (e, lb) in enumerate(edges.items()):
            self.add_edge(idx+1, e[0], e[1], lb)

    def build_graph(self, sentence):
        """Build a graph from raw text (of a sentence)

        Parameters
        ----------
        sentence : iterable
            A list of strings
        """
        nodes = {}
        edges = {}
        for line in sentence:
            eles = line.strip().split('\t')
            if len(eles) != 6:  # TODO: parse corpora of different formats
                continue
            word, pos, type_, idx, hd_idx, rel = eles
            dp_idx, hd_idx = int(idx), int(hd_idx)
            dp_s = '{}/{}'.format(type_, pos)  # TODO: use CorpusFormatter
            nodes[dp_idx] = dp_s
            # we only have the 'to' idx, so might not have the 'to' node string
            # so we add all node strings first and record edges
            edges[(hd_idx, dp_idx)] = rel

        self.generate_graph(nodes, edges)

    def match_feature(self, feature):
        """Match a sentence with a feature (and target pair)"""
        tree_match(self, feature)

    def match_target_feature(self, feature):
        """Match a graph with a path (a tree/graph object).

        Parameters
        ----------
        feature : :class:`FeatureGraph`

        Returns
        -------
        valid matches : iterable
        """
        valid_matches = []
        # iterate over all nodes of graph
        for v, vitem in self.nodes:
            # find the head node
            vitem = vitem.get('label', 'ROOT')
            if not feature.match_node(vitem, idx=feature.target):
                continue
            # start from this node and check if graph matches the path
            valid_nodes = [(v, feature.target)]
            valid_edges = []
            # Breadth search
            match_sub_template(sentence=self, feature=feature, valid_nodes=valid_nodes, valid_edges=valid_edges)
        return feature

    def show(self, v_label='label', e_label='rel', figsize=(5.0, 5.0)):
        draw_tree(self.graph, v_label=v_label, e_label=e_label, figsize=figsize)

    def __repr__(self):
        '''
        output = ""
        for v in self.graph.nodes():
            output += '\nv: {id}'.format(id=v)
            # output += '\nv {id} {label}'.format(id=v, label=l['label'])
        for e1, e2, rel in self.graph.edges(data='rel'):
            output += '\ne: {e1} {e2} {rel}'.format(e1=e1, e2=e2, rel=rel)
        output += '\n'
        '''
        nodes = [None] * self.graph.number_of_nodes()
        for v in self.graph.nodes():
            nodes[v] = self.graph.nodes[v].get('label', 'ROOT')
        return ' '.join(nodes)


def match_sub_template(sentence=None, feature=None, valid_nodes=None, valid_edges=None):
    if len(valid_nodes) == feature.graph.number_of_nodes():
        matched_nodes = {idx: sentence.graph.nodes[vid]['label'] for vid, idx in valid_nodes}
        matched_edges = {(head, tail): it for it, (head, tail) in valid_edges}
        feature.add_match(matched_nodes, matched_edges)
        return

    # iterate all neighbors of current node
    # cur is the last element of valid_seq
    currs, currt = valid_nodes[-1]
    prevs, prevt = valid_nodes[-2] if len(valid_nodes) > 1 else (-1, -1)
    # check predecessors
    for predt in feature.graph.predecessors(currt):  # there is only one
        # check if the neighbor has already been visited
        if predt == prevt:
            continue
        for preds in sentence.graph.predecessors(currs):
            if preds == prevs:
                continue
            srel = sentence.graph[preds][currs]['rel']
            predsitem = sentence.nodes[preds]['label']
            trelpatt = feature.graph[predt][currt]['rel']
            predtpatt = feature.nodes[predt]['label']
            # check if edge and node match
            if re.match(trelpatt, srel) and re.match(predtpatt, predsitem):
                # as match an edge (and 'to' node), append neighbor to valid sequence
                valid_nodes.append((preds, predt))
                valid_edges.append((srel, (predt, currt)))
                # recursively match graph and path
                match_sub_template(sentence=sentence, feature=feature, valid_nodes=valid_nodes,
                                   valid_edges=valid_edges)
                valid_nodes.pop(); valid_edges.pop()
    # check successors
    for succt in feature.graph.successors(currt):
        if succt == prevt:
            continue
        for succs in sentence.graph.successors(currs):
            if succs == prevs:
                continue
            srel = sentence.graph[currs][succs]['rel']  # -> relation label between current node and neighbor
            succsitem = sentence.nodes[succs]['label']
            trelpatt = feature.graph[currt][succt]['rel']
            succtpatt = feature.nodes[succt]['label']
            # check if edge and node match path
            if re.match(trelpatt, srel) and re.match(succtpatt, succsitem):
                # as match an edge (and 'to' node), append neighbor to valid sequence
                valid_nodes.append((succs, succt))
                valid_edges.append((srel, (currt, succt)))
                # recursively match graph and path
                match_sub_template(sentence=sentence, feature=feature, valid_nodes=valid_nodes,
                                   valid_edges=valid_edges)
                valid_nodes.pop(); valid_edges.pop()


class TemplateGraph(DiGraph):
    """Class representing a dependency template tree/graph"""
    def __init__(self, nodes=None, edges=None, graph=None):
        """Construct a dependency template graph by template inputs.

        Parameters
        ----------
        nodes : dict
            mapping from id to node
        edges : dict
            mapping from two-tuple (of nodes) edge to label
        """
        super(TemplateGraph, self).__init__()
        if not graph:
            for idx, v in nodes.items():
                self.add_node(idx, v)
            for idx, (e, lb) in enumerate(edges.items()):
                self.add_edge(idx+1, e[0], e[1], lb)
        else:
            self.graph = graph.copy()

        self.nonlinear = not self.islinear(self.graph)

    @staticmethod
    def islinear(template):
        digx = template
        gx = digx.to_undirected()
        if gx.number_of_edges() == 1:
            return True
        cur = 0
        for v in gx.nodes:
            if gx.degree(v) == 1:
                cur = v
                break
        if cur == 0:
            return False

        nxt = next(gx.neighbors(cur))  # only one element in this iterator
        while True:
            neighbors = list(gx.neighbors(nxt))
            # if digx.in_degree(nxt) == 2:  # a dependent cannot have two heads
            #     return False
            # the two header nodes have one degree, the others have two degree
            if len(neighbors) == 3:
                return False
            if len(neighbors) == 1:  # reach the other header
                break
            for n in neighbors:
                if n != cur:
                    cur, nxt = nxt, n
                    break
        return True

    def match_node(self, item, idx=0):
        m = re.compile(self.graph.nodes[idx]['label']).match(item)
        return m

    def match_edge(self, rel, idx=0):
        m = re.compile(self.graph.edges[idx]['rel']).match(rel)
        return m

    def show(self, v_label='label', e_label='rel', figsize=(5.0, 5.0)):
        draw_tree(self.graph, v_label=v_label, e_label=e_label, figsize=figsize)


class FeatureGraph(TemplateGraph):
    """Class representing a feature graph inherited from the class TemplateGraph.
    So it will have the same structure of the template from which it is generated.
    The generating process of a feature object would be:
    * 1. replicate a (tree) structure of the template
    * 2. set target node index
    * 3. set feature properties for each node (except for the target) and each edge.
         The feature properties (i.e. True or False) would be stored in **attributes** of nodes and edges
    """
    def __init__(self, template, target=-1, feature_filter={}):
        """

        Parameters
        ----------
        template : :class:`~TemplateGraph`
        target : int
            Node index of the template
        feature_filter : dict
            The first dict indicates caring or not the specific types on a node of the template.
            i.e. {2: False}
            Taking the following template as an example (and index 1 as the target):
                { 1: '(\w+/(N))\w*', 2: '(\w+/(V))\w*', 3: '(\w+/(N))\w*' }
                { (2, 1): '\w*(obj)', (2, 3): '\w*(obj)' }
            Then the given dict means that we care about a target type and a feature type
            who co-occur with a same verb both as an object, no matter which specific verb it is.
            For example, target type 'girl/NN' co-occur with the feature 'hd/dobj_*/V_dp/iobj_apple/NN'
            for a number of times.
        """
        super(FeatureGraph, self).__init__(graph=template.graph)
        self.target = target
        self.set_feature(feature_filter)
        self.matched_nodes = []
        self.matched_edges = []
        self.depth = get_depth(self.graph)

    @property
    def size(self):
        return self.graph.number_of_nodes()

    def set_feature(self, feature_filter={}):

        # set properties of each node and edge from the values of template to values of feature
        for v in self.graph.nodes():
            if v == self.target:
                self.graph.nodes[v]['feature'] = False
            elif v in feature_filter:
                self.graph.nodes[v]['feature'] = feature_filter[v]
            else:
                self.graph.nodes[v]['feature'] = True

        for e in self.graph.edges():
            self.graph.edges[e]['feature'] = True

    def set_target(self, target):
        self.target = target

    def add_match(self, matched_nodes, matched_edges):
        """Add matched nodes and edges

        Parameters
        ----------
        matched_nodes : dict
            mapping from node index to item string
        matched_edges : dict
            mapping from edge index to relation string
        """
        self.matched_nodes.append(matched_nodes)
        self.matched_edges.append(matched_edges)

    def show(self, v_label='label', e_label='rel', figsize=(5.0, 5.0)):
        draw_tree(self.graph, v_label=v_label, e_label=e_label, figsize=figsize)

    def show_match(self, index=1, v_label='label', e_label='rel', figsize=(5.0, 5.0)):
        draw_match(self, idx=index, figsize=figsize)
        # draw_tree(self.graph, v_label=v_label, e_label=e_label, figsize=figsize)


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


def tree_match(sentence, feature):
    """Match the sentence with the feature.

    Parameters
    ----------
    sentence : :class:`~SentenceGraph`
    feature : :class:`~FeatureGraph`
    """
    try:
        assert sentence.istree
    except Exception as err:
        logger.error(err, str(sentence))
        return
    assert feature.istree
    # find feature root
    froot = get_root(feature.graph)

    # iterate over each sentence node that matches the root of the feature
    for v, vitem in sentence.nodes:
        vitem = vitem.get('label', 'ROOT')
        if not feature.match_node(vitem, idx=froot):
            continue
        mapping = {froot: v}  # mapping of the first level
        # recursively match the sentence with the feature from this node
        subtree_match(sentence=sentence, feature=feature, lmatches=deque([[mapping]]))


def subtree_match(sentence=None, feature=None, lmatches=None):
    """
    Parameters
    ----------
    sentence : :class:`~SentenceGraph`
    feature : :class:`~FeatureGraph`
    lmatches : queue (collections.deque)
        Contains a list of possible matches.
        Each match is a (finally the length is `feature.depth`) lists of levels of the feature.
        Element example: feature node idx -> sentence node idx.
    """
    # if find a match
    if len(lmatches) <= 0:
        return
    match_depth = len(lmatches[0])
    if match_depth == feature.depth:
        for lmatch in lmatches:
            # merge dicts of levels
            nodemapping = dict()
            for m in lmatch:
                nodemapping.update(m)
            # remap feature node to sentence node label (type)
            matched_nodes = {fn: sentence.nodes[sn]['label'] for fn, sn in nodemapping.items()}
            # remap feature edge (tuple of node idx) to sentence edge label (dependency relation)
            matched_edges = {(head, tail): sentence.edges[(nodemapping[head], nodemapping[tail])]['rel']
                             for head, tail in feature.edges}
            # append matched nodes and edges to the feature
            feature.add_match(matched_nodes, matched_edges)
        return

    size = len(lmatches)  # number of possible matches
    for _ in range(size):
        lmatch = lmatches.popleft()  # get each possible match of all levels
        currmap = lmatch[-1]  # node index mapping of current level
        # match the next level of the sentence based on the matched 'mapping' of current level
        matched, matched_succs = match_level(sentence, feature, currmap)
        if not matched:
            continue
        # append the map of next level to current matches
        # for each possible map of next level, create a new match of previous levels and append the new map
        for sucmap in matched_succs:
            newmap = deepcopy(lmatch)
            newmap.append(sucmap)
            lmatches.append(newmap)

    # recursively match the next level
    subtree_match(sentence=sentence, feature=feature, lmatches=lmatches)


def match_level(sentence, feature, currmap):
    """Match the next level based on the index mapping (feature index -> sentence index) of current level

    Parameters
    ----------
    sentence : :class:`~SentenceGraph`
    feature : :class:`~FeatureGraph`
    currmap : dict
        Index mapping from sentence node to feature node (of current level).
        e.g. feature node idx -> sentence node idx

    Returns
    -------
    A list of dicts: feature node idx -> sentence node idx
    """
    nextmaps = deque([dict()])  # node mapping of next level
    for fn, sn in currmap.items():
        # if reach a leaf of the feature
        if feature.out_degree(fn) == 0:
            # then there is no need to matched its next level (no successors)
            continue
        # matched successors of 'sn' (sentence node) with successors of 'fn' (feature node)
        succmatched, succmaps = match_successors(sentence, sn, feature, fn)
        if succmatched:
            # there might be several matches of (fn, sn)
            size = len(nextmaps)
            for _ in range(size):
                prevone = nextmaps.popleft()
                for onemap in succmaps:
                    newmap = deepcopy(prevone)
                    newmap.update(onemap)
                    nextmaps.append(newmap)
        else:  # there is no match in the next level
            return False, None

    return True, list(nextmaps)


def match_successors(sentence, scur, feature, fcur):
    """Match sentence successors with feature successors based on current sentence node and feature node.

    Parameters
    ----------
    sentence : :class:`~SentenceGraph`
    scur : int
        Current node index of sentence
    feature : :class:`~FeatureGraph`
    fcur : int
        Current node index of feature

    Returns
    -------
    A list of dicts: feature node idx -> sentence node idx
    """
    mapping = deque([dict()])
    # return if reach a leaf
    if feature.out_degree(fcur) == 0:
        return True, mapping

    # we have a list of successors of current sentence node and a list of successors of current feature node
    # we need to match these sentence nodes with feature nodes
    for fsuc in feature.successors(fcur):
        size = len(mapping)
        i = 0
        while i < size:
            onemap = mapping.popleft()
            msuccs = []
            for ssuc in sentence.successors(scur):
                if ssuc in onemap:
                    continue
                vitem = sentence.nodes[ssuc]['label']
                erel = sentence.edges[scur, ssuc]['rel']
                vmatched = feature.match_node(vitem, idx=fsuc)
                ematched = feature.match_edge(erel, idx=(fcur, fsuc))
                if vmatched and ematched:
                    msuccs.append(ssuc)
            if len(msuccs) > 0:
                for ssuc in msuccs:
                    newmap = deepcopy(onemap)
                    newmap[ssuc] = fsuc
                    mapping.append(newmap)
            i += 1

    mapping = [{v: k for k, v in m.items()} for m in mapping]
    return True if len(mapping) > 0 else False, mapping
