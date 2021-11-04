"""
Test dependency relation functions
"""
import pytest
from collections import deque

from nephosem.deprel.basic import SentenceGraph, TemplateGraph, FeatureGraph
from nephosem.deprel.basic import tree_match, subtree_match, match_level, match_successors


@pytest.fixture()
def sent():
    sentence = "The	DT	the	1	2	det\n" \
               "boy	NN	boy	2	3	nsubj\n" \
               "gives	V	give	3	0	ROOT\n" \
               "the	DT	the	4	5	det\n" \
               "girl	NN	girl	5	3	iobj\n" \
               "a	DT	a	6	8	det\n" \
               "green	JJ	green	7	8	amod\n" \
               "apple	NN	apple	8	3	dobj\n" \
               "for	IN	for	9	3	prep\n" \
               "Valentine	NP	Valentine	10	9	pobj\n"
    sent = SentenceGraph(sentence=sentence.split('\n'))
    yield sent


@pytest.fixture()
def feat():
    tplt_nodes = { 1: '(\w+/(N))\w*', 2: '(\w+/(V))\w*', 3: '(\w+/(N))\w*' }
    tplt_edges = { (2, 1): '\w*(obj)', (2, 3): '\w*(obj)' }
    tplt = TemplateGraph(tplt_nodes, tplt_edges)
    feature = FeatureGraph(tplt, target=2)
    yield feature


@pytest.fixture()
def featex():
    tplt_nodes = {1: '(\w+/(N))\w*', 2: '(\w+/(V))\w*', 3: '(\w+/(N))\w*', 4: '(\w+/DT)\w*', 5: '(\w+/JJ)\w*'}
    tplt_edges = {(2, 1): '\w*(obj)', (2, 3): '\w*(obj)', (1, 4): 'det', (3, 5): 'amod'}
    tplt = TemplateGraph(tplt_nodes, tplt_edges)
    feature = FeatureGraph(tplt, target=2)
    yield feature


class TestDepBasic(object):
    def test_tree_match(self, sent, feat):
        tree_match(sent, feat)

    def test_subtree_match(self, sent, feat):
        subtree_match(sent, feat, lmatches=deque([[{2: 3}]]))
        subtree_match(sent, feat, lmatches=deque([[{2: 3}, {1: 5, 3: 8}], [{2: 3}, {1: 8, 3: 5}]]))

    def test_match_level(self, sent, feat):
        matched, mapping = match_level(sent, feat, {1: 5, 3: 8})

    def test_match_successors(self, sent, feat):
        matched, mapping = match_successors(sent, 3, feat, 2)
