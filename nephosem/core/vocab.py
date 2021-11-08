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


"""Vocabulary Class

Usage examples
==============

Initialize a vocabulary with a Python dict e.g.

>>> from nephosem.tests.utils import common_texts
>>> from nephosem import Vocab
>>>
>>>

"""

import os
import re
import operator
import random
import logging
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from six import iteritems

from nephosem import utils

__all__ = ['Vocab']

logger = logging.getLogger(__name__)


class Vocab(object):

    def __init__(self, data=None, encoding='utf-8'):
        """
        Parameters
        ----------
        data : dict, defaultdict, pandas.DataFrame
            The passed data could be a dict, defaultdict or pandas.DataFrame
            The default value is set to None, because an empty Vocab could be created
        encoding : str
            encoding of corpus files, with default 'utf-8'

        """
        self.freq_dict = defaultdict(int)
        self.raw_vocab = None  # not used currently
        self.FILTERPRESENT = False
        self.encoding = encoding  # TODO: do we need this encoding???

        self.freq_dict = self._construct(data)
        # if provided with non empty data or not None, set FILTERPRESENT to True
        if len(self.freq_dict.keys()) > 0:
            self.FILTERPRESENT = True

    @property
    def dataframe(self):
        """Generate dataframe dynamically every time it is called.
        Sort items first by frequency (descending)
        and then by alphabetic ascending order.
        """
        tuples = sorted(self.freq_dict.items(), key=operator.itemgetter(0))
        tuples = sorted(tuples, key=operator.itemgetter(1), reverse=True)
        dataframe = pd.DataFrame(tuples, columns=['item', 'freq'])
        return dataframe

    # ------- construction methods -------
    def _construct(self, data):
        """Construct a Vocab based on the passed data.
        Check the type of input data
        if it is a Python dict: deepcopy a new dict by it
        if it is a pandas.DataFrame: generate frequency dict by dataframe
        """
        if data is None:
            return dict()
        func_dict = {
            dict: deepcopy,
            defaultdict: deepcopy,
            pd.DataFrame: self._gen_freq_dict_by_dataframe,
        }
        dtype = type(data)

        if dtype in func_dict:
            func = func_dict[dtype]
            freq_dict = func(data)  # use corresponding function to construct freq_dict
            return freq_dict
        else:
            raise NotImplementedError("Error: the freq_dict parameter should be a dict or dict filename!")

    @staticmethod
    def _gen_freq_dict_by_dataframe(dataframe):
        """
        This private class method takes a pandas.DataFrame as an input parameter,
        and generate a frequency dictionary (frequency list) based on it.
        """
        items, freqs = dataframe['item'].tolist(), dataframe['freq'].tolist()
        freq_dict = {k: v for k, v in zip(items, freqs)}
        return freq_dict

    # ------- magic methods -------
    def __contains__(self, item):
        return item in self.freq_dict

    def __getitem__(self, arg):
        """Implement this magic method for Vocab class.
        Case 1: vocab['purse/NN0']
            Get the frequency of an item.
        Case 2: vocab[ ['lemma1/pos1', 'lemma2/pos2'] ]
            Select a sub vocab based on a list of items.
        Case 3: vocab[ [2, 3] ]
            Select a sub vocab based on a list of integers.
            The sub vocab has the items whose frequencies are only those in the provided list.
        Case 4: vocab[:10]
            Select a sub vocab. Perform as Python list slicing.
            The list order is 'frequency descending'.
            e.g. Vocab({'a': 1, 'b': 2, 'c': 3})[:2] -> Vocab({'b': 2, 'c': 3})

        Parameters
        ----------
        arg

        Returns
        -------

        """
        if utils.is_string(arg):  # 'arg' is a str: e.g. vocab['argument/NN0']
            return self.freq_dict[arg]
        elif isinstance(arg, list):  # vocab[]
            d = dict()
            if len(arg) > 0:
                if utils.is_string(arg[0]):  # 'arg' is a list of strings
                    return self._select_by_items(arg)  # new Vocab of selected items
                elif isinstance(arg[0], int):
                    return self._select_by_freqs(arg)  # new Vocab of selected items
                else:
                    raise NotImplementedError("Error: unimplemented type {}!".format(type(arg[0])))
            else:
                # TODO: dealing with file encoding
                logger.info("Get an empty list and returns an empty vocab.")
                return self.__class__(d, self.encoding)
        elif isinstance(arg, tuple):  # 'arg' is a tuple
            d = self
            for cond in arg:
                # cond: e.g. vocab.freq >= 2, returns a list of items
                # vocab[cond] returns a sub vocab
                d = d.__getitem__(cond)
            return d
        elif isinstance(arg, slice):  # 'arg' is a slice object
            # the slicing operation works on a 'frequency descending' sorted order
            df = self.dataframe[arg]
            return self.__class__(df, self.encoding)
        else:
            raise NotImplementedError("Error: unimplemented type!")

    def __setitem__(self, key, value):
        self.freq_dict[key] = value

    def __getattr__(self, item):
        # slightly better using:
        # vocab.freq >= 2 other than vocab >= 2
        # so vocab.freq returns itself
        if item == 'freq':
            return self

    def __lt__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            res = [k for k, v in self.freq_dict.items() if v < arg]
            return res
        else:
            raise NotImplementedError("Error: unimplemented type!")

    def __le__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            return [k for k, v in self.freq_dict.items() if v <= arg]
        else:
            raise NotImplementedError("Error: unimplemented type!")

    def __eq__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            return [k for k, v in self.freq_dict.items() if v == arg]

    def __gt__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            return [k for k, v in self.freq_dict.items() if v > arg]
        else:
            raise NotImplementedError("Error: unimplemented type!")

    def __ge__(self, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            return [k for k, v in self.freq_dict.items() if v >= arg]
        else:
            raise NotImplementedError("Error: unimplemented type!")

    def __len__(self):
        return len(self.freq_dict.keys())

    def _select_by_items(self, items):
        """Select a sub vocab by a list of items."""
        items_set = set(items)
        d = {k: v for k, v in self.freq_dict.items() if k in items_set}
        return self.__class__(d, self.encoding)

    def _select_by_freqs(self, freqs):
        """Select a sub vocab by a set of frequencies."""
        freqs_set = set(freqs)
        d = {k: v for k, v in self.freq_dict.items() if v in freqs_set}
        return self.__class__(d, self.encoding)

    # ------- basic methods -------
    def keys(self):
        return list(self.freq_dict.keys())

    def values(self):
        return list(self.freq_dict.values())

    def items(self):
        """Same as Python dict.items()"""
        return list(self.freq_dict.items())

    def get_dict(self):
        # for safety, return a new object
        return deepcopy(self.freq_dict)

    def isEmpty(self):
        return True if self.freq_dict is None or len(self.freq_dict.keys()) == 0 else False

    def setFILTER(self, value):
        self.FILTERPRESENT = value

    def increment(self, key, inc=1):
        """Increment the value of a key by 'inc'."""
        if key in self.freq_dict:
            self.freq_dict[key] += inc
        else:
            self.freq_dict[key] = inc

    def get_item_list(self, sorting='alpha', descending=False):
        """Get a sorted list of items based on a sorting order.
        Calls utils.sort_dict().

        Parameters
        ----------
        sorting : str
            'freq' for frequency order, 'alpha' for alphabetic order.
        descending : bool
            If True, sort dict by descending order of 'sorting'.
            Else, sort dict by ascending order of 'sorting'.

        Returns
        -------
        list :
            sorted list of items in the vocabulary
        """
        sorted_items = utils.sort_dict(self.freq_dict, sorting=sorting, descending=descending)
        return sorted_items

    def select_items(self, word):
        """This method takes a word (or lemma) as input
        and returns a Vocab object.
        Whether the provided word matches the items in vocab or not,
        should depend on the type-format of the items.
        If item is 'lemma/pos', then 'lemma' string should be provided.
        If item is 'word/pos', then 'word' string should be provided.
        """
        new_freq_dict = dict()
        for item, freq in iteritems(self.freq_dict):
            # if the provided string equals to the corresponding part of an item
            if word == item.rsplit('/', 1)[0]:
                new_freq_dict[item] = freq
        return self.__class__(new_freq_dict, encoding=self.encoding)

    def make_type_file(self, type_list, out_fname, encoding='utf-8'):
        """This method could be used in the token level workflow
        for generating a typeSelection file

        Parameters
        ----------
        type_list : a list of types
        out_fname : output file name
        encoding
        """
        type_dict = dict()
        for type_str in type_list:
            if type_str in self.freq_dict:
                freq = self.freq_dict[type_str]
            else:
                freq = 0
            type_dict[type_str] = freq
        utils.save_dict_json(type_dict, out_fname, encoding)

    def subvocab(self, items):
        """Select a sub vocab by a list of items.
        If an item is not in the vocab, its frequency is zero.
        """
        sub_vocab = dict()
        for it in items:
            sub_vocab[it] = self.freq_dict.get(it, 0)
        return self.__class__(sub_vocab)

    # ------- filter methods -------
    @staticmethod
    def regex_item(item, pattern):
        """Match an item by a given regular expression pattern."""
        res = re.match(pattern, item)
        return True if res else False

    def match(self, column_name='item', pattern='.'):
        """Match items by a given regular expression pattern.

        Parameters
        ----------
        pattern : str
            Regular expression pattern
        column_name : str
            'item' or 'freq', normally only use 'item'

        Returns
        -------
        list
        """
        df = self.dataframe
        df = df[df.apply(lambda x: self.regex_item(x[column_name], pattern), axis=1)][column_name]
        return list(df)

    def sum(self):
        """Get total sum of all frequencies.
        Just a slightly better method name.
        """
        return self.sum_freq()

    def sum_freq(self):
        """Get total sum of all frequencies."""
        return sum(self.freq_dict.values())

    def equal(self, vocab2):
        """Check whether two vocabularies are equal."""
        freq_dict1 = self.freq_dict
        freq_dict2 = vocab2.freq_dict
        '''
        if len(freq_dict1.keys()) != len(freq_dict2.keys()):
            return False
        for k in freq_dict1.keys():
            if freq_dict1[k] != freq_dict2[k]:
                return False
        return True
        '''
        return freq_dict1 == freq_dict2  # Python dict can directly do this

    def copy(self):
        """Just to have a better name for deepcopy()."""
        return self.deepcopy()

    def deepcopy(self):
        new_dict = deepcopy(self.freq_dict)
        return Vocab(new_dict, encoding=self.encoding)

    # ------- file methods -------
    def save(self, filename, encoding=None, fmt='json', verbose=True):
        """Save vocabulary to file.

        Parameters
        ----------
        filename : str
        encoding : str
            Encoding format: 'utf-8', 'latin-1' ...
            If not provided, use encoding of *Vocab*.
        fmt : str
            File format: 'json', 'plain'.
            The default file format is 'json'.
            The 'plain' format would save frequency dict in the following format:
                type-string[TAB]frequency
            One type per line.
        verbose : bool
            Show information or not.

        """
        encoding = self.encoding if not encoding else encoding
        if verbose:
            logger.info("Saving frequency list (vocabulary)... (in '{}')".format(encoding))
        args = (self.freq_dict, filename,)
        kwargs = {'encoding': encoding}
        func_dict = {
            'json': utils.save_dict_json,
            'plain': utils.save_dict_plain,
            'txt': utils.save_dict_plain,  # the same as 'plain', just for better understanding
        }
        if fmt in func_dict:
            save_fn = func_dict[fmt]  # get the corresponding function
        else:
            raise KeyError("Does not support this file format.")

        save_fn(*args, **kwargs)  # save dict to file
        if verbose:
            logger.info("Stored in {}".format(filename))

    @classmethod
    def load(cls, filename, encoding='utf-8', fmt='json'):
        """Load vocabulary (frequency list) from file.
        The default file format to load the vocabulary is 'json'.

        Parameters
        ----------
        filename : str
        encoding : str
            'utf-8', 'latin-1', ...
        fmt : str
            'json', 'plain', 'txt' (same as plain)

        Returns
        -------
        :class: `~nephosem.Vocab`
        """
        args = (filename,)
        kwargs = {'encoding': encoding}
        fmt_mapping = {
            'json': utils.load_dict_json,
            'plain': utils.load_dict_plain,
            'txt': utils.load_dict_plain,
        }
        if fmt in fmt_mapping:
            load_fn = fmt_mapping[fmt]
        else:
            raise KeyError("Does not support this file format.")

        freq_dict = load_fn(*args, **kwargs)  # use the corresponding function
        return cls(freq_dict, encoding=encoding)  # return Vocab object

    def describe(self):
        """Give a description of Vocab."""
        # add information to the result of self.dataframe.describe()
        basic_des = "Total items: {}\nTotal freqs: {}".format(len(self.freq_dict.keys()), self.sum_freq())
        description = str(self.dataframe.describe()).split('\n')[1:]
        description.insert(0, basic_des)
        return '\n'.join(description)

    def __repr__(self):
        # TODO: if in Python2, the 'str' type might decode item as ASCII code, then raises error
        items = self.freq_dict.items()
        if len(items) <= 7:
            return '[{}]'.format(','.join(map(str, items)))
        items = sorted(items, key=operator.itemgetter(1), reverse=True)
        return '[{} ... {}]'.format(','.join(map(str, items[:3])),
                                    ','.join(map(str, items[-3:])))

    def __str__(self):
        return self.__repr__()

    # ------- methods used by tokenclouds -------
    def select_subsets(self, specif_words, n=300, method='random', indent=''):
        """select subsets / n appearances
        Here, we select, for each word,
        which n items(appearances) will be retrieved from the corpus.

        Parameters
        ----------
        specif_words
            a list of (specified) words
        n
            number of selected appearance
            if n > the frequency of an item, select all appearances
            else, randomly (default) select n appearances
        method
            selecting methods: 'random', ...
        indent
            indentation
        """
        logger.info("\n{}SELECTING N APPEARANCES OF WORDS...".format(indent))
        word2apps = dict()  # word to appearances mapping
        for sw in specif_words:
            word2apps[sw] = WordDict(sw)

        for sw in specif_words:
            wd = word2apps[sw]
            val = self.freq_dict.get(sw, 0)
            if val == 0:
                wd.selected = set()
                wd.n_sel = 0
            else:
                freq_total = val
                n_sel = min(n, freq_total)  # adjusted n
                wd.n_sel = n_sel  # size of selected subset
                if method == 'random':
                    sample = sorted(random.sample(range(freq_total), n_sel))
                    sample = [i+1 for i in sample]
                    cur_sel = set(sample)
                else:  # RETRIEVAL_METHOD_FIRST assumed
                    cur_sel = set([i+1 for i in range(n_sel)])
                wd.selected = cur_sel
            # next item in 'selected' we need to look for is
            # index in wd.selected (1-based)
            # nr items already found in corpus (selected or not)
            wd.n_found = 0

        return word2apps
