import os
from collections import defaultdict
import networkx as nx


class Sentence(object):
    def __init__(self, id, s):
        self.id = id
        self.text = s
        self.parse()

    def parse(self):
        self.content = []
        for line in self.text.split("\n"):
            if "<" == line[0]:
                continue
            self.content.append(line)

    def getContent(self):
        return self.content


class iGraph(object):
    def __init__(self, sentence=None, id2node=None):
        self.graphx = nx.Graph()
        if sentence:
            self.build_graph(sentence, id2node)

    def add_node(self, v_id, v_label=None):
        if v_label:
            self.graphx.add_node(v_id, label=v_label)
        else:
            self.graphx.add_node(v_id)

    def add_edge(self, e_from_node, e_to_node, e_label):
        self.graphx.add_edge(e_from_node, e_to_node, rel=e_label)

    def build_graph(self, sentence, id2node):
        for line in sentence:
            word, pos, wtype, id, next_id, rel = line.strip().split('\t')
            # not sure about using word for type when type == (unknown)
            self.add_node(id, wtype+'/'+pos if wtype != "(unknown)" else word+'/'+pos)
            if not self.graphx.has_node(next_id):
                self.add_node(next_id, (''))
            # TO CHECK
            if int(next_id) == 0:
                continue
            self.add_edge(id, next_id, rel)

    def __repr__(self):
        output = ""
        for v in self.graphx.nodes_iter():
            output += '\nv: {id}'.format(id=v)
            #output += '\nv {id} {label}'.format(id=v, label=l['label'])
        for e1, e2, rel in self.graphx.edges_iter(data='rel'):
            output += '\ne: {e1} {e2} {rel}'.format(e1=e1, e2=e2, rel=rel)
        output += '\n'
        return output


class CorpusModifier(object):
    def __init__(self):
        self.s2id = dict()
        self.id2s = []
        self.node2id = dict()
        self.id2node = []

    def modify_xml_2_edges(self, filename, edge_file, node_file, sid_file):
        sentences = self.genNodesEdges(filename)
        with open(edge_file, 'w') as eFile:
            for s in sentences:
                eFile.write(s + '\n')
        with open(node_file, 'w') as nFile:
            for n, id in self.node2id.iteritems():
                nFile.write('{}\t{}\n'.format(n, str(id)))
        with open(sid_file, 'w') as sFile:
            for s, id in self.s2id.iteritems():
                sFile.write('{}\t{}\n'.format(s, str(id)))

    def gen_edges_from_sentence(self, sentence):
        innerid2node = ['root']
        for line in sentence:
            word, pos, wtype, id1, id2, rel = line.split('\t')
            node = '{}/{}'.format(wtype if wtype != '(unknown)' else word, pos)
            innerid2node.append(node)
            if node not in self.node2id:
                self.node2id[node] = len(self.node2id)
                self.id2node.append(node)
        ss = []
        for line in sentence:
            word, pos, wtype, id1, id2, rel = line.split('\t')
            if int(id2) == 0:   # find a better way!!!
                continue
            ss.append('{},{},{}'.format(self.node2id[innerid2node[int(id1)]],
                                        rel,
                                        self.node2id[innerid2node[int(id2)]]))
        return ss

    def genNodesEdges(self, filename):
        sentences = self.parseXMLCorpus(filename)
        sid = 0
        for s, sentence in sentences:
            self.s2id[s] = sid
            self.id2s.append(s)
            ss = self.gen_edges_from_sentence(sentence)
            ss = '{}:{}'.format(sid, ';'.join(ss))
            yield ss
            sid += 1

    @classmethod
    def parseXMLCorpus(cls, filename):
        sentences = cls.readSentenceFromFile(filename)
        for sid, sentence in sentences:
            s = Sentence(sid, sentence)
            yield s.id, s.content

    @classmethod
    def readSentenceFromFile(cls, filename):
        with open(filename, 'r') as inFile:
            sid, sentence = "", ""
            while True:
                line = inFile.readline()
                if not line:
                    break
                sentence += line
                if line.startswith("<s"):
                    for item in line.strip().split(' '):
                        tmp = item.split('=')
                        if len(tmp) > 1 and tmp[0]=='id':
                            sid = tmp[-1][1:-1]
                if "</s" == line[:3]:
                    yield sid, sentence.strip()
                    sid, sentence = "", ""

    @classmethod
    def edgeModifier(cls, in_edges):
        sentences = ''
        with open(in_edges) as inFile:
            sentence = []
            sid = -1
            for line in inFile:
                nid1, rel, cur_sid, nid2 = line.strip().split('\t')
                if sid != cur_sid:
                    if len(sentence) > 0:
                        sentences += '{}:{}\n'.format(sid, ';'.join(sentence))
                    sid = cur_sid
                    sentence = []
                sentence.append('{},{},{}'.format(nid1, rel, nid2))
            sentences += '{}:{}'.format(sid, ';'.join(sentence))

        return sentences.strip()

"""
corpusFilename = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/encow14ax03sample.xml"
in_edges = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/spark_test/example_edge.csv"
out_edges = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/spark_test/example_edge_new.csv"
sentences = CorpusModifier.edgeModifier(in_edges)
with open(out_edges, 'w') as outFile:
    outFile.write(sentences)
"""
in_edges_path = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/test_corpus/"
out_edges_path = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/corpus_edges/"
i = 0
for f in os.listdir(in_edges_path):
    if not f.startswith("edge"):
        continue
    with open(out_edges_path + f, 'w') as outFile:
        outFile.write(CorpusModifier.edgeModifier(in_edges_path+f))
    print "\n*******\n%s edge file done...\n*******\n" % i
    i += 1


edgeFilename = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/sample_test/edges.txt"
nodeFilename = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/sample_test/nodes.txt"
sidFilename  = "/home/enzocxt/Projects/QLVL/corp/en/ENCOW/sample_test/sids.txt"
#mod = CorpusModifier()
#mod.modify_xml_2_edges(corpusFilename, edgeFilename, nodeFilename, sidFilename)

