import sys
import time
import operator
import csv
from itertools import islice
from collections import defaultdict
from tqdm import tqdm
from memory_profiler import profile

import networkx as nx
import matplotlib.pyplot as plt


class iNode(object):
    def __init__(self, lemma, pos):
        self.lemma = lemma
        self.pos = pos
        self.edges = []


class iEdge(object):
    def __init__(self, from_node, to_node, rel):
        self.from_node = from_node
        self.to_node = to_node
        self.rel = rel


class Path(list):
    def __init__(self, s):
        super(Path, self).__init__()
        edges = s.split("=")
        for e in edges:
            tokens = e.split(":")
            if len(tokens) != 5:
                raise ValueError("Error: context specifications wrong.")
            fromLemma, fromPOS, rel, toPOS, toLemma = tokens
            e = iEdge(iNode(fromLemma, fromPOS), iNode(toLemma, toPOS), rel)
            self.append(e)


class iGraph(object):
    def __init__(self, corpusName=None, sentence=None):
        self.corpusName = corpusName
        self.graphx = nx.Graph()
        self.edges = {}
        self.nodes = {}
        self.num_of_edges = 0
        self.num_of_nodes = 0
        self.node2id = {}   # word/pos
        if sentence:
            self.build_graph(sentence)

    def add_node(self, v_id, v_label=None):
        if v_label is None:
            self.graphx.add_node(v_id)
        else:
            self.graphx.add_node(v_id, label=v_label)
            self.nodes[int(v_id)] = v_label
        # self.num_of_nodes += 1
        # self.node2id['{}/{}'.format(v_label[0], v_label[1])] = v_id

    def add_node_attr(self, v_id, v_label):
        self.graphx[v_id]['label'] = v_label

    def add_edge(self, e_from_node, e_to_node, e_label):
        self.graphx.add_edge(e_from_node, e_to_node, rel=e_label)
        self.edges[(int(e_from_node), int(e_to_node))] = e_label
        self.num_of_edges += 1

    def build_graph(self, sentence):
        for line in sentence:
            eles = line.strip().split('\t')
            if len(eles) != 6:
                continue
            word, pos, typeStr, id, next_id, rel = eles
            self.add_node(id, (typeStr if typeStr != "(unknown)" else word, pos))
            if not self.graphx.has_node(next_id):
                self.add_node(next_id, ('', ''))
            if int(next_id) == 0:
                continue
            self.add_edge(id, next_id, rel)

    def __repr__(self):
        output = '%s:' % self.corpusName
        for v, l in self.graphx.nodes_iter(data='label'):
            output += '\nv {id} {label}'.format(id=v, label=l['label'])
        for (e1, e2), l in self.edges.items():
            output += '\ne {e1} {e2} {rel}'.format(e1=e1, e2=e2, rel=l)
        output += '\n'
        return output

    def __str__(self):
        return self.__repr__()


class Sentence(object):
    def __init__(self, s):
        self.text = s
        self.parse()

    def parse(self):
        self.content = []
        for line in self.text.split("\n"):
            if line.startswith('<'):
                continue
            self.content.append(line)

    def getContent(self):
        return self.content


def processCorpus(corpusName):
    """Read sentences in file, construct a graph for each sentence, and yield graphs"""
    sentences = readSentenceFromFile(corpusName)

    for s in sentences:
        ss = Sentence(s).getContent()
        G = iGraph(sentence=ss)
        g = nx.Graph()
        mapping = {}
        for node, data in G.graphx.nodes_iter(data=True):
            label = data['label']
            nodeString = '{}/{}'.format(label[0], label[1])
            mapping[node] = nodeString
            g.add_node(nodeString)
        for from_node, to_node, data in G.graphx.edges_iter(data=True):
            rel = data['rel']
            from_str, to_str = mapping[from_node], mapping[to_node]
            g.add_edge(from_str, to_str, rel=rel)
        yield g

class SKV(csv.excel):
    delimiter = "\t"

def save_nodes_2_csv(filename):
    type2id = dict()
    with open(filename) as inFile:
        ind = 0
        for line in inFile:
            typeStr, _ = line.strip().split("\t")
            type2id[typeStr] = ind
            ind += 1

    with open("/home/enzocxt/Projects/QLVL/corp/en/ENCOW/test/csv_node.csv", 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        headers = ['type', 'id']
        writer.writerow(headers)
        rows = sorted(type2id.iteritems(), key=operator.itemgetter(0))
        writer.writerows(rows)


def save_edges_2_csv(graphs, outFilename):
    # csv.register_dialect("SKV", SKV)
    csv_path = outFilename
    i = 0
    # first file
    csvfile = open(csv_path + "edge_{}.csv".format(i), 'wb')
    writer = csv.writer(csvfile, delimiter='\t')
    headers = ['from_node', 'rel', 's_id', 'to_node']
    writer.writerow(headers)

    rows = []
    s_id = 0
    type2id = dict()
    for g in graphs:
        if g.size() <= 1:
            continue
        for from_node, to_node, data in g.edges_iter(data=True):
            rel = data['rel']
            if from_node not in type2id:
                type2id[from_node] = len(type2id)
            if to_node not in type2id:
                type2id[to_node] = len(type2id)
            from_id = type2id[from_node]
            to_id = type2id[to_node]
            rows.append((from_id, rel, s_id, to_id))
            if len(rows) == 1000:
                writer.writerows(rows)
                rows = []
        s_id += 1

        if s_id % 2000000 == 0:
            writer.writerows(rows)
            csvfile.close()
            i += 1
            csvfile = open(csv_path + "edge_{}.csv".format(i), 'wb')
            writer = csv.writer(csvfile, delimiter='\t')
            headers = ['from_node', 'rel', 's_id', 'to_node']
            writer.writerow(headers)
            rows = []
        # if s_id > 10000:
        #     break
    if len(rows) != 0:
        writer.writerows(rows)
    csvfile.close()

    with open("/home/enzocxt/Projects/QLVL/corp/en/ENCOW/spark_test/example_node.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        headers = ['type', 'id']
        writer.writerow(headers)
        rows = sorted(type2id.iteritems(), key=operator.itemgetter(0))
        writer.writerows(rows)


def mergeMultiGraph(graphs):
    """
    merge small graphs of sentences into one graph
    one small graph is a graph of a sentence
    """
    # MG = nx.MultiGraph()
    MG = nx.Graph()
    sent_id = 0
    for G in graphs:
        if G.size() <= 1:
            continue
        for node in G.nodes():
            if not MG.has_node(node):
                MG.add_node(node)
        for from_node, to_node, data in G.edges_iter(data=True):
            relation = data['rel']
            if not MG.has_edge(from_node, to_node):
                rel_dict = dict()
                rel_dict[relation] = set()
                rel_dict[relation].add(sent_id+1)
                MG.add_edge(from_node, to_node, rel=rel_dict)
            else:
                rel_dict = MG[from_node][to_node]['rel']
                if relation in rel_dict:
                    rel_dict[relation].add(sent_id+1)
                else:
                    rel_dict = dict()
                    rel_dict[relation] = set()
                    rel_dict[relation].add(sent_id+1)
                MG[from_node][to_node]['rel'] = rel_dict
        sent_id += 1
    if '/' in MG:
        MG.remove_node('/')
    return MG


if __name__ == '__main__':
    filename = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/spark_test/testsentences.conll"
    graphs = processCorpus(filename)
    #save_edges_2_csv(graphs, "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/spark_test/example.edges")
    #save_nodes_2_csv("/home/enzocxt/Projects/QLVL/corp/en/ENCOW/spark_test/example.nodefreq")
    mg = mergeMultiGraph(graphs)
    #nx.draw(mg, with_labels=True)
    pos = nx.spring_layout(mg)
    nx.draw(mg, pos)
    #node_labels = nx.get_node_attributes(mg, 'id')
    #nx.draw_networkx_labels(mg, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(mg, 'rel')
    nx.draw_networkx_edges(mg, pos, labels=edge_labels)
    plt.show()
