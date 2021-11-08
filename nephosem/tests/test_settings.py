"""
Test functions for basics
"""

import os
import pytest
import pandas as pd
from copy import deepcopy

import nephosem
from nephosem.conf import ConfigLoader
from nephosem.core.terms import CorpusFormatter, ItemNode, TypeNode, TokenNode

rootdir = nephosem.rootdir
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


class TestCorpusFormatter(object):

    def test_line_format(self):
        conf = ConfigLoader()
        base_sett = conf.settings
        base_sett['corpus-path'] = os.path.join(rootdir, 'tests/data/corp/QLVLNewsCorpus/Netherlands')
        base_sett['output-path'] = "{}/data".format(curdir)
        base_sett["file-encoding"] = "latin1"
        base_sett["left-span"] = 4
        base_sett["right-span"] = 4

        # case 1 : "[word]\TAB[lemma]"
        sett1 = deepcopy(base_sett)
        sett1["line-machine"] = "([^\t]+)\t([^\t]+)"
        sett1["separator-line-machine"] = "</arti(kel|cle)>"
        sett1['line-format'] = 'word,lemma'
        # case 1.1
        sett1['type'] = 'lemma'
        sett1['colloc'] = 'lemma'
        sett1['token'] = 'word/fid/lid'
        formatter = CorpusFormatter(sett1)
        line = 'Cat\tcat'
        match = formatter.match_line(line)
        type_ = formatter.get_type(match)
        assert type_ == 'cat'
        colloc = formatter.get_colloc(match)
        assert colloc == 'cat'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'Cat/file1/11'
        # case 1.2
        sett1['type'] = 'lemma'
        sett1['colloc'] = 'word'
        sett1['token'] = 'lemma/fid/lid'
        formatter = CorpusFormatter(sett1)
        line = 'Cat\tcat'
        match = formatter.match_line(line)
        type_ = formatter.get_type(match)
        assert type_ == 'cat'
        colloc = formatter.get_colloc(match)
        assert colloc == 'Cat'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'cat/file1/11'

        # case 2 : "[word]\TAB[lemma]\TAB[pos]"
        sett2 = deepcopy(base_sett)
        sett2["line-machine"] = "([^\t]+)\t([^\t]+)\t([^\t]+)"
        sett2["separator-line-machine"] = "</arti(kel|cle)>"
        line = 'Cat\tcat\tnoun'
        sett2['line-format'] = 'word,lemma,pos'
        # case 2.1
        sett2['type'] = 'lemma/pos'
        sett2['colloc'] = 'lemma/pos'
        sett2['token'] = 'lemma/pos/fid/lid'
        formatter = CorpusFormatter(sett2)
        match = formatter.match_line(line)
        type_ = formatter.get_type(match)
        assert type_ == 'cat/noun'
        colloc = formatter.get_colloc(match)
        assert colloc == 'cat/noun'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'cat/noun/file1/11'
        # case 2.2
        sett2['type'] = 'word'
        sett2['colloc'] = 'lemma/pos'
        sett2['token'] = 'word/fid/lid'
        formatter = CorpusFormatter(sett2)
        match = formatter.match_line(line)
        type_ = formatter.get_type(match)
        assert type_ == 'Cat'
        colloc = formatter.get_colloc(match)
        assert colloc == 'cat/noun'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'Cat/file1/11'

        # case 3 : "[word]\TAB[pos]\TAB[lemma]"
        sett2 = deepcopy(base_sett)
        sett2["line-machine"] = "([^\t]+)\t([^\t]+)\t([^\t]+)"
        sett2["separator-line-machine"] = "</arti(kel|cle)>"
        line = 'Cat\tnoun\tcat'
        # case 3.1
        sett2['line-format'] = 'word,pos,lemma'
        sett2['type'] = 'lemma/pos'
        sett2['colloc'] = 'lemma/pos'
        sett2['token'] = 'word/pos/fid/lid'
        formatter = CorpusFormatter(sett2)
        match = formatter.match_line(line)
        type_ = formatter.get_type(match)
        assert type_ == 'cat/noun'
        colloc = formatter.get_colloc(match)
        assert colloc == 'cat/noun'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'Cat/noun/file1/11'

        # case 3 : "[word]\TAB[lemma]\TAB[pos]\TAB[head]\TAB[rel]\TAB[tail]\TAB[null1]\TAB[null2]"
        sett3 = deepcopy(base_sett)
        sett3["line-machine"] = "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t_\t_"
        sett3["separator-line-machine"] = "</arti(kel|cle)>"
        # line = 'De	de	det	3	det	5	_	_'
        line = 'De\tde\tdet\t3\tdet\t5\t_\t_'
        # case 3.1
        sett3['line-format'] = 'word,lemma,pos,head,rel,tail,null1,null2'
        sett3['type'] = 'lemma/pos'
        sett3['colloc'] = 'lemma/pos'
        sett3['token'] = 'word/pos/fid/lid'
        formatter = CorpusFormatter(sett3)
        match = formatter.match_line(line)
        assert match is not None
        type_ = formatter.get_type(match)
        assert type_ == 'de/det'
        colloc = formatter.get_colloc(match)
        assert colloc == 'de/det'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'De/det/file1/11'
        # case 3.2
        sett3['line-format'] = 'word,lemma,pos'
        sett3['type'] = 'lemma/pos'
        sett3['colloc'] = 'lemma/pos'
        sett3['token'] = 'word/pos/fid/lid'
        formatter = CorpusFormatter(sett3)
        match = formatter.match_line(line)
        assert match is not None
        assert type_ == 'de/det'
        colloc = formatter.get_colloc(match)
        assert colloc == 'de/det'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'De/det/file1/11'

        # case 4 : "[head]\TAB[word]\TAB[lemma]\TAB[pos]\TAB[rel]\TAB[tail]\TAB[null1]\TAB[null2]"
        sett3 = deepcopy(base_sett)
        sett3["line-machine"] = "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t_\t_"
        sett3["separator-line-machine"] = "</arti(kel|cle)>"
        line = '3\tDe\tde\tdet\tdet\t5\t_\t_'
        # case 4.1
        sett3['line-format'] = 'head,word,lemma,pos,rel,tail,null1,null2'
        sett3['type'] = 'lemma/pos'
        sett3['colloc'] = 'lemma/pos'
        sett3['token'] = 'word/pos/fid/lid'
        formatter = CorpusFormatter(sett3)
        match = formatter.match_line(line)
        assert match is not None
        type_ = formatter.get_type(match)
        assert type_ == 'de/det'
        colloc = formatter.get_colloc(match)
        assert colloc == 'de/det'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'De/det/file1/11'

        # case 5 :
        sett5 = deepcopy(base_sett)
        sett5["line-machine"] = "[^\t\n]+\t([^\t]+)\t([^\t]+)\t([^\t]+)\t[^\t]+\t[^\t]+\t_\t_"
        sett5["separator-line-machine"] = "</arti(kel|cle)>"
        line = '1	Herbundeling	her_bundeling	noun	0	ROOT	_	_'
        sett5['line-format'] = 'word,lemma,pos'
        sett5['type'] = 'lemma/pos'
        sett5['colloc'] = 'lemma/pos'
        sett5['token'] = 'word/pos/fid/lid'
        formatter = CorpusFormatter(sett5)
        match = formatter.match_line(line)
        assert match is not None
        type_ = formatter.get_type(match)
        assert type_ == 'her_bundeling/noun'
        colloc = formatter.get_colloc(match)
        assert colloc == 'her_bundeling/noun'
        token = formatter.get_token(match, 'file1', '11')
        assert token == 'Herbundeling/noun/file1/11'
