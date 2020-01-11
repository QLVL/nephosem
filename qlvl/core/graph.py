#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: QLVL <qlvl@kuleuven.be>
# Copyright (C) 2018 QLVL KULeuven

import logging
import re
import operator
from collections import deque, defaultdict
from copy import deepcopy

import pandas as pd

import networkx as nx
import xml
from xml.dom import minidom

from qlvl.specutils.deputils import match_graph, tree_match, get_depth, draw_tree, draw_match, parse_pattern

logger = logging.getLogger(__name__)


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


class DiGraph(object):
    """
    Attributes
    ----------
    graph : :class:`~network.DiGraph`
    """
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
        try:
            istree = nx.is_tree(self.graph)
            return istree
        except:
            return False

    def out_degree(self, v):
        return self.graph.out_degree(v)

    def in_degree(self, v):
        return self.graph.in_degree(v)

    def successors(self, v):
        return self.graph.successors(v)

    def predcessors(self, v):
        return self.graph.predecessors(v)

    def add_node(self, v_id, **kwargs):
        """Add a node with id and label (optional) to graph."""
        self.graph.add_node(v_id, **kwargs)

    def add_edge(self, e_id, from_v, to_v, **kwargs):
        """Add an edge to graph."""
        self.graph.add_edge(from_v, to_v, id=e_id, **kwargs)

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
    def __init__(self, nodes=None, edges=None, sentence=None, formatter=None, settings=None):
        """Construct a graph by a sentence and id2node.
        When the nodes and edges are provided, directly create a sentence graph based on them.
        When the nodes and edges are not provided, create a sentence graph based the corpus sentence string lines.

        Parameters
        ----------
        nodes : dict, optional
        edges : dict, optional
        sentence : iterable
            A list of strings (each string is a line in the corpus file).
            Each string line has the same format as the `formatter` indicates.
        formatter : :class:`~qlvl.core.terms.CorpusFormatter`
        """
        super(SentenceGraph, self).__init__()
        self.formatter = formatter
        if nodes and edges:
            self.generate_graph(nodes, edges)
        elif sentence:
            self.build_graph(sentence)
        else:
            raise ValueError("Please provide a sentence!")

    def generate_graph(self, nodes, edges):
        for idx, vals in nodes.items():
            self.add_node(idx, **vals)
        for idx, (e, lb) in enumerate(edges.items()):
            self.add_edge(idx+1, e[0], e[1], **lb)

    def build_graph(self, sentence):
        """Build a graph from raw text (of a sentence)

        Parameters
        ----------
        sentence : iterable
            A list of strings
        """
        nodes = defaultdict(lambda: defaultdict(str))
        edges = defaultdict(lambda: defaultdict(str))
        for line in sentence:
            line = line.strip()
            match = self.formatter.match_line(line)
            if match is None:
                continue
            node_attr = self.formatter.node_attr.split(',')
            edge_attr = self.formatter.edge_attr.split(',')
            # get column names of current index and head index
            currID_column = self.formatter.settings.get('currID', 'ID')
            headID_column = self.formatter.settings.get('headID', 'HEAD')
            node_idx = int(self.formatter.get(match, currID_column))
            head_idx = int(self.formatter.get(match, headID_column))
            for col in self.formatter.global_columns:
                val = self.formatter.get(match, col)
                if col in node_attr:
                    nodes[node_idx][col] = val
                elif col in edge_attr:
                    edges[(head_idx, node_idx)][col] = val

        self.generate_graph(nodes, edges)

    def match_pattern(self, macro):
        """Match a sentence with a feature pattern (and target pair).
        Append the results to the attribute lists `matched_nodes` and `matched_edges` of the feature object.
        A `matched node` is a dict mapping node index to type string.
        A `matched edge` is a dict mapping edge index to dependency relation.
        e.g. one match would be:
            * nodes: `{1: 'boy/NN', 2: 'give/V', 3: 'girl/NN'}`
            * edges: `{(2, 1): 'nsubj', (2, 3): 'iobj'}`

        Parameters
        ----------
        macro : :class:`~qlvl.core.graph.MacroGraph`
            The feature pattern to be matched to the sentence.
        """
        return tree_match(self, macro)

    def match_target_feature(self, feature):
        """Match a graph with a path (a tree/graph object).

        Parameters
        ----------
        feature : :class:`MacroGraph`

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


class PatternGraph(DiGraph):
    connector = '/'

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
        super(PatternGraph, self).__init__()
        if not graph:
            for idx, attrs in nodes.items():
                if isinstance(attrs, str):
                    # parse the regex string into subgroups
                    subs = attrs.rsplit(self.connector, 1)
                    attrs = {}
                    for reg in subs:
                        start = reg.index('?P<')
                        end = reg.index('>')
                        attr = reg[start + 3:end]
                        attrs[attr] = reg.replace(reg[start:end+1], '')
                self.add_node(idx, **attrs)
            for idx, (e, attrs) in enumerate(edges.items()):
                if isinstance(attrs, str):
                    subs = attrs.rsplit(self.connector, 1)
                    attrs = {}
                    for reg in subs:
                        start = reg.index('?P<')
                        end = reg.index('>')
                        attr = reg[start + 3:end]
                        attrs[attr] = reg.replace(reg[start:end+1], '')
                self.add_edge(idx+1, e[0], e[1], **attrs)
        else:
            self.graph = graph.copy()

        self.repr_ = None
        self.nonlinear = not self.islinear(self.graph)

        self.node_repr_fmt = "LEMMA/POS"
        self.edge_repr_fmt = "DEPREL"

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

    def match_node(self, node, idx=0):
        """Match the passing sentence node with the corresponding node in the pattern graph.

        Parameters
        ----------
        node : dict
            A dict of node attributes of the sentence graph.
            e.g. {'FORM': 'The', 'POS': 'DT', 'LEMMA': 'the'}
        idx : int
            An integer index of the corresponding node of the pattern graph.
        """
        if not node or len(node) == 0:
            return None

        match = {}
        # for each attribute of pattern node
        # match the corresponding sentence node
        for attr, regex in self.graph.nodes[idx].items():
            m = re.match(regex, node[attr])
            if m is None:
                return None
            match[attr] = m
        return match

    def match_edge(self, edge, idx=0):
        """Match the passing sentence edge with the corresponding edge in the pattern graph.

        Parameters
        ----------
        edge : dict
            A dict of edge attributes of the sentence graph.
            For most cases, there is only one attribute `DEPREL`.
        idx : int or tuple of int
            An integer index of the corresponding edge of the pattern graph.
        """
        match = {}
        for attr, regex in self.graph.edges[idx].items():
            if attr == 'id':  # skip the default attribute `id`
                continue
            m = re.match(regex, edge[attr])
            if m is None:
                return None
            match[attr] = m
        return match

    def show(self, v_label='label', e_label='rel', figsize=(5.0, 5.0)):
        draw_tree(self.graph, v_label=v_label, e_label=e_label, figsize=figsize)

    @classmethod
    def read_csv(cls, fname,
                 target_colname='Target Regex', feature_colname='Feature Regex',
                 sep='\t', header=0, **kwargs):
        """Read feature patterns from a CSV/TSV file.
        This method uses `pandas.read_csv()`. If there is any reading error, please refer to the documentation of pandas.

        Parameters
        ----------
        fname : str
            Filename of feature patterns in Kris' notation.
        target_colname : str, default 'Target Regex'
            Column name of the target regular expression in the CSV/TSV file.
        feature_colname : str, default 'Feature Regex'
            Column name of the feature regular expression in the CSV/TSV file.
        sep : str, default '\t'
            Delimiter to use.

        Returns
        -------

        """
        # read patterns from CSV/TSV file
        feat_df = pd.read_csv(fname, sep=sep, header=header, **kwargs)
        # pd.options.display.max_colwidth = 100
        # feat_df[['ID', 'Target Regex', 'Feature Regex']].head(30)
        templates = []
        # process them to construct template objects
        for i in range(feat_df.shape[0]):
            target = feat_df[target_colname].iloc[i].strip()
            feature = feat_df[feature_colname].iloc[i].strip()
            # feature = feature.replace('(nsubj)', '^nsubj$')
            node_dict, edge_dict = parse_pattern(target, feature)
            tplt = cls(node_dict, edge_dict)
            # add the original string representation as an attribute of template
            tplt.repr_ = feature
            templates.append(tplt)

        return templates

    @classmethod
    def read_graphml(cls, fname):
        with open(fname) as fin:
            graphs = []
            header = []
            ender = None
            graphml = []
            line = fin.readline()
            while line:
                sline = line.strip()
                if sline.startswith('<graph') and not sline.startswith('<graphml'):
                    break
                header.append(line)
                line = fin.readline()
            while line:
                sline = line.strip()
                if sline.startswith('</graphml'):
                    ender = line
                    break
                graphml.append(line)
                if sline.startswith('</graph'):
                    graphs.append(graphml)
                    graphml = []
                # start of a graph
                line = fin.readline()
        graphs = [''.join(header + gml + [ender]) for gml in graphs]

        doc = minidom.parse(fname)
        attrs = doc.getElementsByTagName('key')
        defaults = {}
        for attr in attrs:
            attrname = attr.attributes['attr.name'].value
            try:
                default = attr.getElementsByTagName('default')[0]
                defaults[attrname] = default.childNodes[0].data
            except:
                pass

        patterns = []
        for gmlstring in graphs:
            g = nx.parse_graphml(graphml_string=gmlstring, node_type=int)
            # add default attribute values
            for n, attrs in g.nodes(data=True):
                for attr, val in defaults.items():
                    if attr not in attrs:
                        attrs[attr] = val

            gxml = minidom.parseString(gmlstring)
            gid = gxml.getElementsByTagName('graph')[0].attributes['id'].value
            p = cls(graph=g)
            p.id = int(gid)
            patterns.append(p)
        return patterns

    def __repr__(self):
        return 'pattern {}'.format(self.id)


class MacroGraph(PatternGraph):
    """Class representing a feature graph inherited from the class PatternGraph.
    So it will have the same structure of the template from which it is generated.
    The generating process of a feature object would be:
    * 1. replicate a (tree) structure of the template
    * 2. set target node index
    * 3. set feature properties for each node (except for the target) and each edge.
         The feature properties (i.e. True or False) would be stored in **attributes** of nodes and edges
    """
    connector = '/'

    def __init__(self, pattern, target_idx=-1, feature_idx=-1, target_filter={}, feature_filter={}):
        """

        Parameters
        ----------
        pattern : :class:`~PatternGraph`
        target_idx : int
            Node index of the template
        target_filter : :class:`~qlvl.core.vocab.Vocab`, optional
            When given, matched sentences which satisfy the given targets.
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
        super(MacroGraph, self).__init__(graph=pattern.graph)
        self.target_idx = target_idx
        # example: {'LEMMA': 1, 'POS': 1}
        self.target_node_attrs = {}
        self.feature_idx = feature_idx
        # example: {'LEMMA': 1, 'POS': 1}
        self.feature_node_attrs = {}
        # example: {'DEPREL': 1}
        self.feature_edge_attrs = {}
        self.target_filter = target_filter
        self.repr_ = pattern.repr_
        # self.set_feature(feature_filter)
        self.matched_nodes = []
        self.matched_edges = []
        self.depth = get_depth(self.graph)

    @property
    def size(self):
        return self.graph.number_of_nodes()

    def target(self, index=0):
        """Return the target of the `index` match."""
        target_idx = self.target_idx
        vals = []
        # for attr in self.node_repr_fmt.split(self.connector):
        for attr, gid in self.target_node_attrs.items():
            target_regex = self.graph.nodes[target_idx][attr]
            target_node = self.matched_nodes[index][target_idx][attr]
            mtarget = re.match(target_regex, target_node)
            try:
                mtrgt_repr = mtarget.group(gid)
            except:
                mtrgt_repr = '*'  # ''
            vals.append(mtrgt_repr)
        # return '/'.join(mtarget.groups())
        return self.connector.join(vals)

    def feature(self, index=0):
        """Transform a matched node and edge to a feature string"""
        node_dict = self.matched_nodes[index]
        edge_dict = self.matched_edges[index]
        # preorder traversal, start from the root
        start = 1
        reprs = []
        # add the string repr of the start node
        if start == self.target_idx:
            node_repr = '#T#'
        else:
            node_attrs = node_dict[start]
            node_repr = self.get_node_repr(start, node_attrs)
        reprs.append(node_repr)
        # recursively traverse the sub-trees of the successors
        self.preorder_recur(node_dict, edge_dict, curr=start, reprs=reprs)
        return ''.join(reprs)

    def preorder_recur(self, node_dict, edge_dict, curr=0, reprs=None):
        # the curr node repr has been add to `reprs`
        num_succs = self.graph.out_degree(curr)
        if num_succs > 1:
            reprs.append('->')
            reprs.append('[')  # add '[' to feature string
            # get all (edge and) node representations of the successors
            succs = []
            for succ in self.graph.successors(curr):
                edge_attrs = edge_dict[(curr, succ)]
                edge_repr = self.get_edge_repr((curr, succ), edge_attrs)
                node_attrs = node_dict[succ]
                node_repr = self.get_node_repr(succ, node_attrs)
                succs.append((succ, ':'.join([edge_repr, node_repr])))
            # sort the representations
            succs = sorted(succs, key=operator.itemgetter(1))
            for succ, srepr in succs:
                if succ == self.target_idx:
                    # replace the target node repr by '#T#'
                    reprs.append(':'.join([srepr.split(':')[0], '#T#']))
                else:
                    reprs.append(srepr)
                self.preorder_recur(node_dict, edge_dict, curr=succ, reprs=reprs)
                reprs.append(',')
            reprs.pop()  # pop the last ','
            reprs.append(']')  # add ']' to feature string
        elif num_succs == 1:
            reprs.append('->')
            # not add '[]'
            succ = list(self.graph.successors(curr))[0]
            edge_attrs = edge_dict[(curr, succ)]
            edge_repr = self.get_edge_repr((curr, succ), edge_attrs)
            node_attrs = node_dict[succ]
            node_repr = self.get_node_repr(succ, node_attrs)
            if succ == self.target_idx:
                # replace the target node repr by '#T#'
                reprs.append(':'.join([edge_repr, '#T#']))
            else:
                reprs.append(':'.join([edge_repr, node_repr]))
            self.preorder_recur(node_dict, edge_dict, curr=succ, reprs=reprs)

    def get_node_repr(self, nid, node_attrs):
        """Get representation of a (matched) node."""
        vals = []
        for attr, gid in self.feature_node_attrs.items():
            node_regex = self.graph.nodes[nid][attr]
            mnode = re.match(node_regex, node_attrs[attr])
            try:
                mnode_repr = mnode.group(gid)
            except:
                mnode_repr = '*'  # ''
            vals.append(mnode_repr)
        return self.connector.join(vals)

    def get_edge_repr(self, eid, edge_attrs):
        """Get representation of an (matched) edge."""
        vals = []
        for attr, gid in self.feature_edge_attrs.items():
            edge_regex = self.graph.edges[eid][attr]
            medge = re.match(edge_regex, edge_attrs[attr])
            try:
                medge_repr = medge.group(gid)
            except:
                medge_repr = '*'  # ''
            vals.append(medge_repr)
        return self.connector.join(vals)

    def feature_old(self, index=0):
        """Deprecated"""
        node_dict = deepcopy(self.matched_nodes[index])
        node_dict.pop(1, None)
        for idx, node in node_dict.items():
            vals = []
            for attr in self.node_repr_fmt.split(self.connector):
                node_regex = self.graph.nodes[idx][attr]
                mnode = re.match(node_regex, node[attr])
                try:
                    mnode_repr = mnode.group(1)
                except:
                    mnode_repr = ''
                vals.append(mnode_repr)
            node_dict[idx] = self.connector.join(vals)

        edge_dict = self.matched_edges[index]
        for idx, edge in edge_dict.items():
            vals = []
            for attr in self.edge_repr_fmt.split(self.connector):
                edge_regex = self.graph.edges[idx][attr]
                medge = re.match(edge_regex, edge[attr])
                try:
                    medge_repr = medge.group(1)
                except:
                    medge_repr = ''
                vals.append(medge_repr)
            edge_dict[idx] = self.connector.join(vals)

        node_items = sorted(node_dict.items(), key=operator.itemgetter(0))
        matched_nodes = [n for _, n in node_items]
        matched_edges = list(edge_dict.values())
        # TODO: for case
        # >[(agent) by/IN>[pobj (\w+)/(N)\w*],(nsubjpass) (\w+)/(N)\w*]
        # the order of relations is not correct
        '''
        {2: 'by/IN', 3: 'boy/NN', 4: 'apple/NN'}
        {(1, 2): 'agent', (1, 4): 'nsubjpass', (2, 3): ''}
        '''
        special_patt = '>[(agent) by/IN>[pobj (\w+)/(N)\w*],(nsubjpass) (\w+)/(N)\w*]'
        if self.repr_ == special_patt:
            matched_edges = ['agent', '', 'nsubjpass']

        matched = [None] * (len(matched_nodes) + len(matched_edges))
        matched[::2] = matched_edges
        matched[1::2] = matched_nodes

        eles = []
        parts = self.repr_.split(',')
        for part in parts:
            for c in ['[', ']', '<-', '->']:
                part = part.replace(c, ' ')
            eles.extend(part.split())

        specials = self.repr_
        for e in eles:
            specials = specials.replace(e, '|')
        specials = specials.replace(' ', '_')
        specials = specials.split('|')

        mrepr = [None] * (len(specials) + len(matched))
        mrepr[::2] = specials
        mrepr[1::2] = matched
        mrepr = ''.join(mrepr)
        return mrepr

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

    @classmethod
    def read_csv(cls, fname,
                 target_colname='Target Regex', feature_colname='Feature Regex',
                 sep='\t', header=0, **kwargs):
        templates = PatternGraph.read_csv(fname, target_colname=target_colname, feature_colname=feature_colname,
                                          sep=sep, header=header, **kwargs)
        # target node index is always 1
        features = [cls(tplt, target=1) for tplt in templates]
        return features

    @classmethod
    def read_xml(cls, fname, patterns):
        """Example XML:
        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <target-feature-list>
            <target-fmt>
                <node-fmt>
                    <LEMMA group="1"/>
                    <POS group="1"/>
                    <string connector="/">LEMMA/POS</string>
                </node-fmt>
            </target-fmt>
            <feature-fmt>
                <node-fmt>
                    <LEMMA group="1"/>
                    <POS group="1"/>
                    <string connector="/">LEMMA/POS</string>
                </node-fmt>
                <edge-fmt>DEPREL</edge-fmt>
            </feature-fmt>
            <target-feature-macro id="1">
                ...
            </target-feature-macro>
            ...
        </target-feature-list>
        ```

        Parameters
        ----------
        fname : str
        patterns : list, of :class:`~qlvl.core.graph.PatternGraph`

        Returns
        -------
        list, of :class:`~qlvl.core.graph.MacroGraph`
        """
        id2patt = {p.id: p for p in patterns}
        doc = minidom.parse(fname)

        # parse the target node format
        target_fmt = doc.getElementsByTagName('target-fmt')[0]
        target_node_fmt = target_fmt.getElementsByTagName('node-fmt')[0]
        target_str = target_node_fmt.getElementsByTagName('string')[0].childNodes[0].data  # e.g. 'LEMMA/POS'
        connector = target_str.attributes['connector'].value  # commonly use '/'
        target_attrs = target_str.split(connector)  # e.g. ['LEMMA', 'POS']
        # compare attribute names
        trgtattr2group = {}
        for attr in target_attrs:
            group_id = target_node_fmt.getElementsByTagName(attr)[0].attributes['group'].value
            trgtattr2group[attr] = int(group_id)

        # parse the feature node and edge formats
        feature_fmt = doc.getElementsByTagName('feature-fmt')[0]
        feature_node_fmt = feature_fmt.getElementsByTagName('node-fmt')[0]
        feature_node_str = feature_node_fmt.getElementsByTagName('string')[0].childNodes[0].data  # e.g. 'LEMMA/POS'
        connector = feature_node_str.attributes['connector'].value  # '/'
        feature_node_attrs = feature_node_str.split(connector)  # e.g. ['LEMMA', 'POS']
        # compare attribute names
        featattr2group = {}
        for attr in feature_node_attrs:
            group_id = feature_node_fmt.getElementsByTagName(attr)[0].attributes['group'].value
            featattr2group[attr] = int(group_id)
        feature_edge_fmt = feature_fmt.getElementsByTagName('edge-fmt')[0].childNodes[0].data  # e.g. 'DEPREL'

        # parse macros
        macros = []
        macro_xmls = doc.getElementsByTagName('target-feature-macro')
        for macroxml in macro_xmls:
            macro = cls.parse_macro_xml(macroxml, id2patt)
            macro.target_node_attrs = trgtattr2group
            macro.feature_node_attrs = featattr2group
            macro.feature_edge_attrs = {feature_edge_fmt: 1}
            macros.append(macro)

        return macros

    @classmethod
    def parse_macro_xml(cls, macroxml, id2patt):
        """Example XML:
        ```xml
        <target-feature-macro id="1">
            <sub-graph-pattern id="1"/>
            <target nodeID="2">
                <description>Empty</description>
            </target>
            <feature nodeID="1">
                <description>Words that depend directly on the target.</description>
            </feature>
        </target-feature-macro>
        ```

        Parameters
        ----------
        macroxml : :class:`~xml.dom.minidom.Element`
            XML element of a macro
        id2patt : dict
            The dict mapping index to pattern

        Returns
        -------
        :class: `~qlvl.core.graph.MacroGraph`
        """
        # get macro id
        macro_id = int(macroxml.attributes['id'].value)
        # get pattern id
        patt = macroxml.getElementsByTagName('sub-graph-pattern')[0]
        pattern_id = int(patt.attributes['id'].value)
        # parse the target information
        target = macroxml.getElementsByTagName('target')[0]
        target_idx = int(target.attributes['nodeID'].value)
        target_desc_tag = target.getElementsByTagName('description')
        target_desc = target_desc_tag[0].childNodes[0].data if len(target_desc_tag) > 0 else "no description"
        # parse the feature information
        feature = macroxml.getElementsByTagName('feature')[0]
        try:
            feature_idx = int(feature.attributes['featureID'].value)
        except Exception as e:
            feature_idx = -1
        feat_desc_tag = feature.getElementsByTagName('description')
        feat_desc = feat_desc_tag[0].childNodes[0].data if len(feat_desc_tag) > 0 else "no description"
        # construct a macro
        macro = cls(id2patt[pattern_id], target_idx=target_idx, feature_idx=feature_idx)
        macro.id = macro_id
        macro.target_desc = target_desc
        macro.feature_desc = feat_desc
        return macro

    def __repr__(self):
        # if self.repr_:
        #     return self.repr_
        # else:
        #     return super(MacroGraph, self).__repr__()
        return "macro {}".format(self.id)
