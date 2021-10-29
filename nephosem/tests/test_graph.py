"""
Test functions for dependency graph
"""

import os
import pytest
import pandas as pd
from copy import deepcopy

import qlvl
from qlvl.conf import ConfigLoader
from qlvl.core.terms import CorpusFormatter
from qlvl.deprel.basic import SentenceGraph, TemplateGraph

rootdir = qlvl.rootdir
curdir = os.path.dirname(os.path.realpath(__file__))


'''
# Test sentence:
The boy gives the girl a green apple for Valentine

# conll format:
The	DT	the	1	2	amod
boy	NN	boy	2	3	nsubj
gives	V	give	3	0	ROOT
the	DT	the	4	5	amod
girl	NN	girl	5	3	dobj
a	DT	a	6	8	amod
green	JJ	green	7	8	amod
apple	NN	apple	8	3	dobj
for	IN	for 9	10	amod
Valentine	NP	Valentine	10	8	amod

'''


@pytest.fixture()
def orgsent():
    sent = "The	DT	the	1	2	amod\n" \
           "boy	NN	boy	2	3	nsubj\n" \
           "gives	V	give	3	0	ROOT\n" \
           "the	DT	the	4	5	amod\n" \
           "girl	NN	girl	5	3	dobj\n" \
           "a	DT	a	6	8	amod\n" \
           "green	JJ	green	7	8	amod\n" \
           "apple	NN	apple	8	3	dobj\n" \
           "for	IN	for 9	10	amod\n" \
           "Valentine	NP	Valentine	10	8	amod\n"
    yield sent


@pytest.fixture()
def exsent():
    # example sentence
    # The boy gives the girl a green apple for Valentine
    v_labels = {
        1: 'the/DT',
        2: 'boy/NN',
        3: 'give/V',
        4: 'the/DT',
        5: 'girl/NN',
        6: 'a/DT',
        7: 'green/JJ',
        8: 'apple/NN',
        9: 'for/IN',
        10: 'Valentine/NP',
    }

    e_labels = {
        (2,1): 'amod',
        (3,2): 'nsubj',
        (5,4): 'amod',
        (3,5): 'dobj',
        (8,6): 'amod',
        (8,7): 'amod',
        (3,8): 'dobj',
        (10,9): 'amod',
        (8,10): 'amod',
    }
    yield (v_labels, e_labels)


class TestTemplateGraph(object):
    def test_islinear(self):
        # test cases of templates
        # 1
        tplt1_nodes = {1: '(\w+/(N))\w*', 2: '(\w+/(V))\w*'}
        tplt1_edges = {(2, 1): 'nsubj'}
        tplt1_g = TemplateGraph(tplt1_nodes, tplt1_edges)
        assert not tplt1_g.nonlinear

        # 2
        tplt2_nodes = {1: '(\w+/(N))\w*', 2: '(\w+/(V))\w*', 3: '(\w+/(N))\w*'}
        tplt2_edges = {(2, 1): 'nsubj', (2, 3): 'dobj'}
        tplt2_g = TemplateGraph(tplt2_nodes, tplt2_edges)
        assert not tplt2_g.nonlinear

        # 3
        tplt3_nodes = {1: '(\w+/(V))\w*', 2: '(\w+/(N))\w*', 3: '(\w+/jj)\w*'}
        tplt3_edges = {(1, 2): 'nsubj', (2, 3): 'amod'}
        tplt3_g = TemplateGraph(tplt3_nodes, tplt3_edges)
        assert not tplt3_g.nonlinear

        # 4
        tplt4_nodes = {1: '(\w+/(N))\w*', 2: '(\w+/(V))\w*', 3: '(\w+/(N))\w*', 4: '(\w+/(N))\w*'}
        tplt4_edges = {(2, 1): 'nsubj', (2, 3): 'dobj', (2, 4): 'dobj'}
        tplt4_g = TemplateGraph(tplt4_nodes, tplt4_edges)
        assert not tplt4_g.nonlinear

        # 5 incorrect template
        tplt5_nodes = {1: '(\w+/(V))\w*', 2: '(\w+/(N))\w*', 3: '(\w+/jj)\w*'}
        tplt5_edges = {(1, 2): 'nsubj', (3, 2): 'amod'}  # incorrect direction
        tplt5_g = TemplateGraph(tplt5_nodes, tplt5_edges)
        assert not tplt5_g.nonlinear

class TestSentenceGraph(object):
    def test_init(self, orgsent, exsent):
        sent_g = SentenceGraph(sentence=orgsent.split('\n'))
        sent_g = SentenceGraph(nodes=exsent[0], edges=exsent[1])
