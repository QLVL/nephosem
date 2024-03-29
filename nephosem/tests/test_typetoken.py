#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import pytest

import nephosem
from nephosem.conf import ConfigLoader
from nephosem.tests.utils import datapath
from nephosem.models.typetoken import ItemFreqHandler, ColFreqHandler
from nephosem.models.typetoken import TypeToken


curdir = os.path.dirname(__file__)

freq_dict = {'the/DT': 53, 'girl/NNS': 2, 'look/VBZ': 5, 'healthy/JJ': 10, 'boy/NN': 22, 'at/IN': 6, 'girl/NN': 19, 'as/IN': 1, 'she/PRP': 2, 'eat/VBZ': 10, 'less/RBR': 1, 'food/NN': 4, 'all/PDT': 1, 'old/JJ': 3, 'boy/NNS': 3, 'be/VBD': 2, 'give/VBN': 3, 'an/DT': 3, 'apple/NN': 14, 'old/JJR': 1, 'a/DT': 6, 'in/IN': 1, 'house/NN': 1, 'apple/NNS': 5, 'be/VBP': 4, 'for/IN': 1, 'be/VBZ': 4, 'eat/VBN': 4, 'by/IN': 3, 'very/RB': 1, 'apples/NNP': 2, ',/,': 3, 'without/IN': 1, 'look/VBG': 1, 'sit/VBZ': 2, 'down/RP': 1, 'about/RB': 1, 'ten/JJ': 1, 'say/VBZ': 1, 'that/IN': 1, 'should/MD': 2, 'eat/VB': 7, 'that/WDT': 1, 'baby/NN': 2, 'have/VBZ': 2, 'her/PRP': 1, 'and/CC': 5, 'tasty/JJ': 5, 'sit/VBD': 1, 'on/IN': 1, 'look/VBD': 2, "'s/POS": 1, 'both/DT': 1, 'what/WP': 2, 'give/VBZ': 2, 'not/RB': 2, 'do/VBZ': 1, 'ten/CD': 1, 'year/NNS': 1, 'ask/VBD': 1, 'about/IN': 1, 'eat/VBG': 1, 'during/IN': 1, 'year/NN': 2, 'like/VBZ': 1, 'to/TO': 2, 'love/VBZ': 1, 'which/WDT': 1, 'give/VBD': 1, 'him/PRP': 1, 'this/DT': 1, 'be/VBN': 1}


@pytest.fixture()
def settings():
    conf = ConfigLoader()
    settings = conf.settings
    settings['corpus-name'] = 'StanfDepSents'
    settings['corpus-path'] = datapath('StanfDepSents')
    settings['output-path'] = datapath('')
    settings['line-machine'] = '([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)'
    settings['line-format'] = 'word,pos,lemma,tail,head,rel'
    settings['type'] = 'lemma/pos'
    settings['colloc'] = 'lemma/pos'
    settings['token'] = 'word/pos/fid/lid'
    yield settings


@pytest.fixture()
def settings_word():
    conf = ConfigLoader()
    settings = conf.settings
    settings['corpus-name'] = 'StanfDepSents'
    settings['corpus-path'] = datapath('StanfDepSents')
    settings['output-path'] = datapath('')
    settings['line-machine'] = '([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)'
    settings['line-format'] = 'word,pos,lemma,tail,head,rel'
    settings['type'] = 'word'
    settings['colloc'] = 'lemma/pos'
    settings['token'] = 'word/fid/lid'
    yield settings


class TestTypeToken(object):
    def test_build_vocab(self, settings, settings_word):
        # since every time the module uses the tmp folder, it cleans it
        # so if we use the same tmp folder for several functions
        # the clean operation of one function would remove the tmp files generated by another function
        # therefore, we use different tmp folders for different functions
        dtime = datetime.datetime.now().strftime('%Y-%m-%d+%H:%M:%S')
        settings['tmp-path'] = os.path.join(nephosem.tmpdir, dtime)
        ifhan = ItemFreqHandler(settings)
        vocab1 = ifhan.build_item_freq()

        dtime = datetime.datetime.now().strftime('%Y-%m-%d+%H:%M:%S')
        settings['tmp-path'] = os.path.join(nephosem.tmpdir, dtime)
        ifhan = ItemFreqHandler(settings)
        vocab2 = ifhan.build_item_freq(fnames=datapath('StanfDepSents.fnames'))
        assert vocab1.equal(vocab2)
        assert vocab1 is not vocab2

        dtime = datetime.datetime.now().strftime('%Y-%m-%d+%H:%M:%S')
        settings_word['tmp-path'] = os.path.join(nephosem.tmpdir, dtime)
        ifhan = ItemFreqHandler(settings_word)
        vocab3 = ifhan.build_item_freq()

        dtime = datetime.datetime.now().strftime('%Y-%m-%d+%H:%M:%S')
        settings_word['tmp-path'] = os.path.join(nephosem.tmpdir, dtime)
        ifhan = ItemFreqHandler(settings_word)
        vocab4 = ifhan.build_item_freq(fnames=datapath('StanfDepSents.fnames'))
        assert vocab3.equal(vocab4)
        assert vocab3 is not vocab4

    def test_build_col_freq(self, settings):
        return
        tthan = TypeToken(corpus_name='test_brown', settings=settings)
        vocab = tthan.build_vocab()
        colfreq1 = tthan.build_col_freq(fnames=datapath('Brown.fnames'), row_vocab=vocab)
        vocab1 = vocab[vocab.freq > 2]
        vocab2 = vocab[vocab.freq > 1]
        colfreq2 = tthan.build_col_freq(fnames=datapath('Brown.fnames'), row_vocab=vocab1, col_vocab=vocab2)
        print(colfreq1)
        print(colfreq2)
        # assert False
