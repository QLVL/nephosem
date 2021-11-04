"""
Test Vocab Class
"""

import os
import codecs
import json
import pytest
import pandas as pd

from nephosem import Vocab

curdir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def vocab():
    d = {
        'Abc/NP0': 3,
        'acbed/NN0': 2,
        'bcd/NN0': 2,
        'bcd/NN2': 1,
        'cdef/VVB': 4,
        'cdef/VVN': 4,
    }
    yield Vocab(d)

# TODO: more test cases


def array_equal(array1, array2):
    """the two arrays need not be the same order"""
    if not isinstance(array1, list):
        array1 = list(array1)
    if not isinstance(array2, list):
        array2 = list(array2)
    for e1, e2 in zip(sorted(array1), sorted(array2)):
        if e1 != e2:
            return False
    return True


class TestVocab(object):
    def test_init(self, vocab):
        """Test initialization of Vocab class."""
        d = vocab.get_dict()

        # test input argument: dict
        tmp_vocab = Vocab(d)
        # python equal operator does not only check reference equal
        assert d == tmp_vocab.freq_dict
        assert isinstance(tmp_vocab.dataframe, pd.DataFrame)
        assert tmp_vocab.dataframe.shape[0] == len(tmp_vocab.items())

        # test input keyward argument: dict
        tmp_vocab = Vocab(data=d)
        assert d == tmp_vocab.freq_dict
        assert isinstance(tmp_vocab.dataframe, pd.DataFrame)
        assert tmp_vocab.dataframe.shape[0] == len(tmp_vocab.items())

        # test input argument: pd.DataFrame
        df = pd.DataFrame(list(d.items()), columns=['item', 'freq'])
        tmp_vocab = Vocab(df)
        assert d == tmp_vocab.freq_dict
        assert isinstance(tmp_vocab.dataframe, pd.DataFrame)
        assert tmp_vocab.dataframe.shape[0] == len(tmp_vocab.items())

        # test input keyward argument: pd.DataFrame
        tmp_vocab = Vocab(data=df)
        assert d == tmp_vocab.freq_dict
        assert isinstance(tmp_vocab.dataframe, pd.DataFrame)
        assert tmp_vocab.dataframe.shape[0] == len(tmp_vocab.items())

    def test_keys(self, vocab):
        freq_dict = vocab.get_dict()
        assert array_equal(freq_dict.keys(), vocab.keys())

    def test_values(self, vocab):
        freq_dict = vocab.get_dict()
        assert array_equal(freq_dict.values(), vocab.values())

    def test_items(self, vocab):
        freq_dict = vocab.get_dict()
        assert array_equal(freq_dict.items(), vocab.items())

    def test_operators(self, vocab):
        assert isinstance(vocab > 3, list)
        assert array_equal(vocab > 3, ['cdef/VVB', 'cdef/VVN'])
        assert array_equal(vocab.freq > 3, ['cdef/VVB', 'cdef/VVN'])
        assert isinstance(vocab < 3, list)
        assert array_equal(vocab < 3, ['bcd/NN0', 'bcd/NN2', 'acbed/NN0'])
        assert array_equal(vocab.freq < 3, ['bcd/NN0', 'bcd/NN2', 'acbed/NN0'])
        assert isinstance(vocab >= 3, list)
        assert array_equal(vocab >= 3, ['Abc/NP0', 'cdef/VVB', 'cdef/VVN'])
        assert array_equal(vocab.freq >= 3, ['Abc/NP0', 'cdef/VVB', 'cdef/VVN'])
        assert isinstance(vocab <= 3, list)
        assert array_equal(vocab <= 3, ['bcd/NN0', 'bcd/NN2', 'acbed/NN0', 'Abc/NP0'])
        assert array_equal(vocab.freq <= 3, ['bcd/NN0', 'bcd/NN2', 'acbed/NN0', 'Abc/NP0'])

    def test_getitem(self, vocab):
        """Test vocab[arg] for different cases."""
        # test arg -> str
        assert vocab['Abc/NP0'] == 3
        # test arg -> list of str
        items = ['bcd/NN0', 'bcd/NN2']
        tmp_vocab = vocab[items]
        assert array_equal(tmp_vocab.keys(), items)
        # test arg -> list of int
        freqs = [2, 3]
        tmp_vocab = vocab[freqs]
        assert array_equal(list(set(tmp_vocab.values())), freqs)  # select items which only have frequency of 2 or 3
        # test arg -> tuple (of conditions)
        tmp_vocab = vocab[vocab.freq >= 2, vocab.freq < 4]
        assert array_equal(tmp_vocab.keys(), ['acbed/NN0', 'bcd/NN0', 'Abc/NP0'])

    def test_increment(self, vocab):
        vocab = vocab.copy()
        vocab.increment('bcd/NN2', inc=1)
        assert vocab['bcd/NN2'] == 2
        vocab.increment('bcd/NN2', inc=2)
        assert vocab['bcd/NN2'] == 4
        vocab.increment('bcd/NN2', inc=-1)
        assert vocab['bcd/NN2'] == 3
        vocab.increment('bcd/NN2', inc=-2)
        assert vocab['bcd/NN2'] == 1

    def test_get_item_list(self, vocab):
        alpha_items_asc = ['Abc/NP0', 'acbed/NN0', 'bcd/NN0', 'bcd/NN2', 'cdef/VVB', 'cdef/VVN']
        assert alpha_items_asc == vocab.get_item_list(sorting='alpha', descending=False)

        alpha_items_des = ['cdef/VVN', 'cdef/VVB', 'bcd/NN2', 'bcd/NN0', 'acbed/NN0', 'Abc/NP0']
        assert alpha_items_des == vocab.get_item_list(sorting='alpha', descending=True)

        freq_items_des = ['cdef/VVB', 'cdef/VVN', 'Abc/NP0', 'acbed/NN0', 'bcd/NN0', 'bcd/NN2']
        assert freq_items_des == vocab.get_item_list(sorting='freq', descending=True)

        freq_items_asc = ['bcd/NN2', 'acbed/NN0', 'bcd/NN0', 'Abc/NP0', 'cdef/VVB', 'cdef/VVN']
        assert freq_items_asc == vocab.get_item_list(sorting='freq', descending=False)

    def test_subvocab(self, vocab):
        tmp_vocab = vocab.subvocab(['bcd/NN0', 'bcd/NN2'])
        assert 'bcd/NN0' in tmp_vocab
        assert 'bcd/NN0' in tmp_vocab
        assert len(tmp_vocab) == 2

    def test_regex_item(self, vocab):
        assert vocab.regex_item('bcd/NN0', 'bcd/')

    def test_match(self, vocab):
        matched_items = vocab.match('item', 'bcd')
        assert array_equal(matched_items, ['bcd/NN0', 'bcd/NN2'])
        # TODO: test more patterns

    def test_sum(self, vocab):
        assert vocab.sum() == 16

    def test_equal(self, vocab):
        tmp_vocab = Vocab(vocab.get_dict())
        assert vocab.equal(tmp_vocab)
        tmp_vocab = Vocab(vocab.get_dict(), encoding='latin-1')  # same object even different encoding format
        assert vocab.equal(tmp_vocab)
        tmp_vocab['acbed/NN0'] = 10
        assert not vocab.equal(tmp_vocab)

    def test_copy(self, vocab):
        tmp_vocab = Vocab(vocab.get_dict())
        assert tmp_vocab.equal(vocab.copy())

    def test_save(self, vocab):
        datadir = "{}/data".format(curdir)
        filename = os.path.join(datadir, 'test.vocab')
        vocab.save(filename, fmt='json')  # default
        vocab.save(filename, fmt='plain')  # plain/txt format
        vocab.save(filename, encoding='latin-1')  # 'latin-1' encoding

    def test_load(self, vocab):
        datadir = "{}/data".format(curdir)
        # test saving as 'json' format
        filename = os.path.join(datadir, 'test.in.vocab')
        with codecs.open(filename, 'w') as fout:
            json.dump(vocab.get_dict(), fout, ensure_ascii=False, indent=4)
        new_vocab = Vocab.load(filename)
        assert new_vocab.equal(vocab)

        # test saving as plain/txt format
        with codecs.open(filename, 'w') as fout:
            for k, v in vocab.items():
                fout.write('{}\t{}\n'.format(k, v))
        new_vocab = Vocab.load(filename, fmt='plain')
        assert new_vocab.equal(vocab)
