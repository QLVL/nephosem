"""
Test functions for basics
"""

import os
import pytest
import pandas as pd
from copy import deepcopy

import qlvl
from qlvl.conf import ConfigLoader
from qlvl.core.terms import CorpusFormatter, ItemNode, TypeNode, TokenNode

rootdir = qlvl.rootdir
curdir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def corpdir():
    corpdir_1 = os.path.join(rootdir, 'tests/data/corp/Brown_wpr-art')
    corpdir_2 = os.path.join(rootdir, 'tests/data/corp/QLVLNewsCorpus/Netherlands')
    yield [corpdir_1, corpdir_2]


@pytest.fixture()
def datadir():
    curdir = os.path.dirname(os.path.realpath(__file__))
    yield '{}/data'.format(curdir)


@pytest.fixture()
def settings():
    conf = ConfigLoader()
    settings = conf.settings
    settings['corpus-path'] = os.path.join(rootdir, 'tests/data/corp/QLVLNewsCorpus/Netherlands')
    settings['output-path'] = "{}/data".format(curdir)
    settings["file-encoding"] = "latin1"
    settings["left-span"] = 4
    settings["right-span"] = 4
    settings["line-machine"] = "([^\t]+)\t([^\t]+)"
    settings["separator-line-machine"] = "</arti(kel|cle)>"
    settings['line-format'] = 'word,lemma'
    settings['type'] = 'lemma'
    settings['token'] = 'word/fid/lid'
    yield settings


class TestItemNode(object):
    def test_init(self):
        conf = ConfigLoader()
        settings = conf.settings
        settings['line-machine'] = "([^\t]+)\t([^\t]+)\t([^\t]+)"
        settings["separator-line-machine"] = "</arti(kel|cle)>"
        settings['line-format'] = 'word,pos,lemma'
        settings['type'] = 'lemma/pos'
        settings['colloc'] = 'lemma/pos'
        settings['token'] = 'word/pos/fid/lid'

        exline = "is	VVB	be"  # example line of corpus

        # case 1 : match -> <regex match>, formatter -> CorpusFormatter object
        formatter = CorpusFormatter(settings)
        m = formatter.match_line(exline)
        itnode = ItemNode(match=m, formatter=formatter)
        assert itnode.word == 'is'
        assert itnode.lemma == 'be'
        assert itnode.pos == 'VVB'
        assert itnode.to_type() == 'be/VVB'
        assert itnode.to_colloc() == 'be/VVB'

        # case 2 : word -> 'is', lemma -> 'be', pos -> 'verb' (at least provide one)
        itnode = ItemNode(word='is', lemma='be', pos='VVB')
        assert itnode.to_type(type_fmt='lemma/pos') == 'be/VVB'
        assert itnode.to_colloc(colloc_fmt='word/pos') == 'is/VVB'


class TestTypeNode(object):
    def test_init(self):
        conf = ConfigLoader()
        settings = conf.settings
        settings['line-machine'] = "([^\t]+)\t([^\t]+)\t([^\t]+)"
        settings["separator-line-machine"] = "</arti(kel|cle)>"
        settings['line-format'] = 'word,pos,lemma'
        settings['type'] = 'lemma/pos'
        settings['colloc'] = 'lemma/pos'
        settings['token'] = 'word/pos/fid/lid'

        exline = "is	VVB	be"  # example line of corpus

        # case 1 : match -> <regex match>, formatter -> CorpusFormatter object
        formatter = CorpusFormatter(settings)
        m = formatter.match_line(exline)
        tpnode = TypeNode(match=m, formatter=formatter)
        assert tpnode.word == 'is'
        assert tpnode.lemma == 'be'
        assert tpnode.pos == 'VVB'
        assert tpnode.type == 'be/VVB'

        # case 2 : type-string -> 'be/verb', type-format -> 'lemma/pos'
        tpnode = TypeNode(type_fmt='lemma/pos', type_str='be/VVB')
        assert tpnode.lemma == 'be'
        assert tpnode.pos == 'VVB'
        assert tpnode.type == 'be/VVB'

        # case 3 : word -> 'is', lemma -> 'be', pos -> 'verb' (at least provide one)
        tpnode = TypeNode(lemma='be', pos='VVB', type_fmt='lemma/pos')
        assert tpnode.type == 'be/VVB'

    def test_get_collocs(self):
        #tknode1 = TokenNode()
        #tpnode = TypeNode(lemma='be', pos='VVB', type_fmt='lemma/pos')
        pass

    def test_sample(self):
        pass

    def test_merge(self):
        pass


class TestTokenNode(object):
    def test_init(self):
        conf = ConfigLoader()
        settings = conf.settings
        settings['line-machine'] = "([^\t]+)\t([^\t]+)\t([^\t]+)"
        settings["separator-line-machine"] = "</arti(kel|cle)>"
        settings['line-format'] = 'word,pos,lemma'
        settings['type'] = 'lemma/pos'
        settings['colloc'] = 'lemma/pos'
        settings['token'] = 'word/pos/fid/lid'

        exline = "is	VVB	be"  # example line of corpus

        # case 1 : match -> <regex match>, formatter -> CorpusFormatter object
        formatter = CorpusFormatter(settings)
        m = formatter.match_line(exline)
        tpnode = TokenNode(match=m, formatter=formatter)
        assert tpnode.word == 'is'
        assert tpnode.lemma == 'be'
        assert tpnode.pos == 'VVB'
        assert tpnode.token == 'is/VVB/unknown/-1'

        # case 2 : token-string -> 'is/verb', token-format -> 'word/pos/fid/lid'
        tpnode = TokenNode(token_fmt='word/pos/fid/lid', token_str='is/VVB/fname/1')
        assert tpnode.word == 'is'
        assert tpnode.pos == 'VVB'
        assert tpnode.token == 'is/VVB/fname/1'

        # case 2 : word -> 'is', lemma -> 'be', pos -> 'verb' (at least provide one)
        tpnode = TokenNode(word='is', pos='VVB', token_fmt='word/pos/fid/lid')
        assert tpnode.token == 'is/VVB/unknown/-1'


class TestWindow(object):
    def test_init(self):
        pass
