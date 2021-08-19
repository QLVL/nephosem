#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: QLVL <qlvl@kuleuven.be>
# Copyright (C) 2018 QLVL KULeuven

import os
import re
import codecs
import json
import operator
import random
import logging

from copy import deepcopy
from collections import defaultdict, deque

from six import iteritems

from qlvl import utils

__all__ = ['Window', 'Getter', 'CorpusFormatter', 'ItemNode', 'TypeNode', 'TokenNode']

logger = logging.getLogger(__name__)


class ItemNode(object):
    """This class represents an item node parsed by `line-machine` regular expression.
    The parsed item node consists of 'word', 'lemma' and 'pos' (if a file line has them).
    """
    connector = '/'

    def __init__(self, match=None, formatter=None,
                 word=None, lemma=None, pos=None,
                 type_fmt=None, colloc_fmt=None, **kwargs):
        """
        Case 1: match -> <regex match>, formatter -> CorpusFormatter object
        Case 2: word -> 'is', lemma -> 'be', pos -> 'verb' (at least provide one)

        Parameters
        ----------
        match : regular expression match object
        formatter : :class:`~qlvl.CorpusFormatter`
        word : str
        lemma : str
        pos : str
        type_fmt : str
            type format string
        colloc_fmt : str
            collocate format string
        fid : str, optional
            file name / id
        lid : int, optional
            line number
        """
        if match and formatter:
            word = formatter.get_word(match)
            lemma = formatter.get_lemma(match)
            pos = formatter.get_pos(match)
        # if TypeNode and TokenNode both have to deal with 'word', 'lemma', 'pos'
        # then put these parts in super class ItemNode
        self.word = word
        self.lemma = lemma
        self.pos = pos

        if formatter:
            type_fmt = formatter.settings['type']
            colloc_fmt = formatter.settings['colloc']
        self.type_fmt = type_fmt
        self.colloc_fmt = colloc_fmt
        self.__dict__.update(**kwargs)

    def to_type(self, type_fmt=None):
        """Get a type string based on the item node.

        Parameters
        ----------
        type_fmt : str
            type format string, i.e. 'lemma/pos'
        """
        type_fmt = type_fmt if type_fmt else self.type_fmt
        if type_fmt:
            return self.connector.join(self.__dict__[c] for c in type_fmt.split(self.connector))
        else:
            raise ValueError("Cannot transform to type from this item!\n"
                             "Please provide type format")

    def to_colloc(self, colloc_fmt=None):
        """Get a collocate string based on the item node.

        Parameters
        ----------
        colloc_fmt : str
            colloc format string, i.e. 'lemma/pos'
        """
        colloc_fmt = colloc_fmt if colloc_fmt else self.colloc_fmt
        if colloc_fmt:
            return self.connector.join(self.__dict__[c] for c in colloc_fmt.split(self.connector))
        else:
            raise ValueError("Cannot transform to colloc from this item!\n"
                             "Please provide colloc format")

    def __str__(self):
        """String representation, like it in the original file line.
        [word]\TAB[pos]\TAB[lemma]
        """
        return '\t'.join([self.word, self.pos, self.lemma])

    def __repr__(self):
        return self.__str__()


class TypeNode(ItemNode):
    """This Class represents a type node which in the token level contains all its token appearances.

    The followings are some important attributes.
    Attributes
    ----------
    lemma : str
    pos : str
    tokens : a list of appearances / tokens
        an appearance includes the token and its collocate types
    """
    connector = '/'

    def __init__(self, match=None, formatter=None, type_fmt=None,
                 type_str=None, word=None, lemma=None, pos=None,
                 tokens=None, **kwargs):
        """Initialize a TypeNode by necessary info.
        Case 1: match -> <regex match>, formatter -> CorpusFormatter object
        Case 2: type-string -> 'be/verb', type-format -> 'lemma/pos'
        Case 3: word -> 'is', lemma -> 'be', pos -> 'verb', type-format -> 'word/pos'

        Parameters
        ----------
        match : regular expression match object
        formatter : :class:`~qlvl.CorpusFormatter`
        type_fmt : str
            type format string
        type_str : str
            type str (i.e. 'lemma/pos')
        word : str
        lemma : str
        pos : str
        tokens : iterable of :class:`~qlvl.TokenNode`
            A list of tokens.
        kwargs
        """
        if match and formatter:  # case 1
            type_fmt = formatter.type_format
            super(TypeNode, self).__init__(match=match, formatter=formatter)
        elif type_fmt:
            if type_str:  # case 2
                eles = type_str.split(self.connector)
                fmts = type_fmt.split(self.connector)
                if len(eles) != len(fmts):
                    raise ValueError("Provided type string and type_fmt length inconsistent!")
                for ele, fmt in zip(eles, fmts):
                    if fmt == 'word':
                        word = ele
                    elif fmt == 'lemma':
                        lemma = ele
                    elif fmt == 'pos':
                        pos = ele
                    else:
                        raise ValueError("Type format should only include 'word', 'lemma', 'pos'!")
                super(TypeNode, self).__init__(word=word, lemma=lemma, pos=pos)
            elif word or lemma or pos:  # case 3: at least provide one
                super(TypeNode, self).__init__(word=word, lemma=lemma, pos=pos)
        else:
            # raise ValueError("Please provide necessary info!")
            raise ValueError("Please provide type format!")

        self.format = type_fmt
        self.tokens = [] if not tokens else deepcopy(tokens)
        self.__dict__.update(**kwargs)

    @property
    def type(self):
        comps = self.format.split(self.connector)
        return self.connector.join([self.__dict__[c] for c in comps])

    @property
    def freq(self):
        """frequency of the type"""
        return len(self.tokens)

    @property
    def collocs(self):
        return self.get_collocs()

    def append_token(self, token):
        self.tokens.append(token)

    def get_collocs(self):
        """Get the collocates of all tokens"""
        collocs = set()
        for tok in self.tokens:
            collocs.update(set(tok.lcollocs))
            collocs.update(set(tok.rcollocs))
        return collocs

    def save(self, filename, fmt='json', encoding='utf-8', verbose=True):
        if verbose:
            print("Saving tokens of {}/{}".format(self.lemma, self.pos))
        data = dict()
        data['lemma'] = self.lemma
        data['pos'] = self.pos
        data['tokens'] = [tok.json_data for tok in self.tokens]
        with codecs.open(filename, 'w', encoding=encoding) as outf:
            json.dump(data, outf, indent=4)
        if verbose:
            print("Stored in file:\n\t{}".format(filename))

    @classmethod
    def load(cls, filename, encoding='utf-8'):
        with codecs.open(filename, 'r', encoding=encoding) as inf:
            data = json.load(inf)
        tokens = [TokenNode.gen_token_from_json_data(tok) for tok in data['tokens']]
        return cls(data['lemma'], data['pos'], tokens=tokens)

    def sample(self, n=300, method='random'):
        """Select n tokens/appearances from all.

        Parameters
        ----------
        n : int
            default is 300
        method : str
            'random', ...

        Returns
        -------
        A new TypeNode object
        """
        if method == 'random':
            sample_indices = set(random.sample(range(self.freq), n))
            sample_tokens = [h for i, h in enumerate(self.tokens) if i in sample_indices]
        else:  # RETRIEVAL_METHOD_FIRST assumed
            sample_tokens = self.tokens[:n]

        wn = TypeNode(type_str=self.type, type_fmt=self.type_fmt, tokens=sample_tokens)
        return wn

    @classmethod
    def merge(cls, tns):
        """Merge a list of TypeNode instances into one

        Parameters
        ----------
        tns : a list of TypeNode instances
        """
        if len(tns) == 0:
            return
        type_str = str(tns[0])
        type_fmt = tns[0].format
        tokens = []
        for tn in tns:
            if str(tn) != type_str or type_fmt != tn.format:
                raise ValueError("TypeNode list has inconsistent type!\n"
                                 "{}".format(str(tn)))
            tokens.extend(tn.tokens)

        # TODO: maybe sort tokens
        return cls(type_str=type_str, type_fmt=type_fmt, tokens=tokens)

    def __str__(self):
        return self.type

    def __repr__(self):
        return self.__str__()


class TokenNode(ItemNode):
    """A TokenNode normal has its left and right context/collocate ItemNodes.

    Attributes
    ----------
    connector : str
        For connecting 'word', 'lemma', 'pos', 'fid' and 'lid'.
        default '/'
    lcollocs : a list of left collocates
    rcollocs : a list of right collocates
        one collocates is an ItemNode object
    """
    connector = '/'

    def __init__(self, token_str=None, token_fmt=None,
                 match=None, formatter=None,  fid='unknown', lid='-1',
                 word=None, pos=None, lemma=None,
                 lcollocs=None, rcollocs=None, **kwargs):
        """
        Case 1: match -> <regex match>, formatter -> CorpusFormatter object
        Case 2: token-string -> 'is/verb/fname/1', token-format -> 'word/pos/fid/lid'
        Case 3: word -> 'is', lemma -> 'be', pos -> 'verb', token-format -> 'word/pos/fid/lid'

        Parameters
        ----------
        fid : str
            file name/id
        lid : str or int
            line number in corpus file
        match :
            Regular expression match object of a corpus line
        formatter : :class:`~qlvl.CorpusFormatter`
        word
        pos
        lemma
        lcollocs : iterable
            A list of left collocates (ItemNode)
        rcollocs : iterable
            A list of right collocates (ItemNode)
        kwargs
        """
        if match and formatter:  # case 1
            token_fmt = formatter.token_format
            super(TokenNode, self).__init__(match=match, formatter=formatter)
        elif token_fmt:
            if token_str:  # case 2
                eles = token_str.split(self.connector)
                fmts = token_fmt.split(self.connector)
                if len(eles) != len(fmts):
                    raise ValueError("Provided type string and type_fmt length inconsistent!")
                for ele, fmt in zip(eles, fmts):
                    if fmt == 'word':
                        word = ele
                    elif fmt == 'lemma':
                        lemma = ele
                    elif fmt == 'pos':
                        pos = ele
                    elif fmt == 'fid':
                        fid = ele
                    elif fmt == 'lid':
                        lid = ele
                    else:
                        raise ValueError("Type format should only include 'word', 'lemma', 'pos'!")
                super(TokenNode, self).__init__(word=word, lemma=lemma, pos=pos)
            elif word or lemma or pos:  # case 3: at least provide one
                super(TokenNode, self).__init__(word=word, lemma=lemma, pos=pos)
        else:
            # raise ValueError("Please provide necessary info!")
            raise ValueError("Please provide token format!")

        self.fid = fid
        self.lid = str(lid)
        self.format = token_fmt
        self.lcollocs = lcollocs
        self.rcollocs = rcollocs
        self.__dict__.update(**kwargs)

    @property
    def lspan(self):
        return len(self.lcollocs)

    @property
    def rspan(self):
        return len(self.rcollocs)

    @property
    def token(self):
        comps = self.format.split(self.connector)
        return self.connector.join([self.__dict__[c] for c in comps])

    @property
    def json_data(self):
        data = dict()
        data['word'] = self.word
        data['pos'] = self.pos
        data['lemma'] = self.lemma
        data['fid'] = self.fid
        data['lid'] = self.lid
        # data['token'] = self.connector.join([self.word, self.pos, self.fid, str(self.lid)])
        data['left-collocates'] = [str(it) for it in self.lcollocs]
        data['right-collocates'] = [str(it) for it in self.rcollocs]
        return data

    @classmethod
    def gen_token_from_json_data(cls, json_data):
        word = json_data['word']
        pos = json_data['pos']
        lemma = json_data['lemma']
        fid = json_data['fid']
        lid = json_data['lid']
        lcollocs = [ItemNode(item=col.split('\t')) for col in json_data['left-collocates']]
        rcollocs = [ItemNode(item=col.split('\t')) for col in json_data['right-collocates']]
        return cls(word, pos, lemma, fid, lid, lcollocs=lcollocs, rcollocs=rcollocs)

    def __str__(self):
        return self.token

    def __repr__(self):
        return self.__str__()


class Window(object):
    """
    Attributes
    ----------
    left_span : int
        left span, window size of left collocates
    right_span : int
        right span, window size of right collocates
    left : deque
        left window
    right : deque
        right window
    node
        center node
    """

    def __init__(self, lspan=10, rspan=10):
        self.left_span = lspan
        self.right_span = rspan
        self.left = self.init_span(lspan)
        self.right = self.init_span(rspan)
        self.node = None

    @staticmethod
    def init_span(size):
        win = deque(maxlen=size)
        for i in range(size):
            win.append(None)
        return win

    def update(self, cur):
        """
        current window: [l1, ...]        [node]  [r1, ...]
        update cur =>   [l2, ..., node]  [r1]    [r2, ..., cur]

        Parameters
        ----------
        cur

        """
        if self.left_span > 0:
            # first, append the node to the left window
            self.left.append(self.node)
        else:  # left window size zero
            pass
        if self.right_span > 0:
            # second, set the node to the first one of the right window
            self.node = self.right[0]
            # third, append the cur to the right window
            self.right.append(cur)
        else:  # right window size zero
            self.node = cur

    def __repr__(self):
        return "{}, {}, {}".format(','.join(map(str, list(self.left))),
                                   self.node,
                                   ','.join(map(str, list(self.right))))

    def __str__(self):
        return self.__repr__()


class Getter(object):

    def __init__(self, settings):
        self.form = settings['form']
        self.match = None
        self._token_line_machine = re.compile(settings['token-line-machine'])
        self._word_line_machine = re.compile(settings['word-line-machine'])
        # self.colloc_line_machine = re.compile(settings['colloc-line-machine'])
        self._separator_line_machine = re.compile(settings['separator-line-machine'])
        self._left_bound_machine = re.compile(settings['left-boundary-machine'])
        self._right_bound_machine = re.compile(settings['right-boundary-machine'])
        self._single_bound_machine = re.compile(settings['single-boundary-machine'])
        # self.init_machine(line)

        self.get_word = self.get_func(settings['get-word'])
        self.get_pos = self.get_func(settings['get-pos'])
        self.get_lemma = self.get_func(settings['get-lemma'])
        self.get_colloc = self.get_func(settings['get-colloc'])
        # self.get_token = self.get_func(settings['get-token'])
        self.get_node = self.get_func(settings['get-' + self.form])

    def init_machine(self, line):
        self.match = self._word_line_machine.match(line)

    def word_line_machine(self, line):
        return self._word_line_machine.match(line)

    def token_line_machine(self, line, fid, lid):
        newline = '\t'.join(line.split('\t').extend([fid, lid]))
        match = self._token_line_machine.match(newline)
        return match

    def separator_line_machine(self, line):
        return self._separator_line_machine.match(line)

    def left_bound_machine(self, line):
        return self._left_bound_machine.match(line)

    def right_bound_machine(self, line):
        return self._right_bound_machine.match(line)

    def single_bound_machine(self, line):
        return self._single_bound_machine.match(line)

    @property
    def word(self):
        return self.get_word(self.match)

    @property
    def pos(self):
        return self.get_pos(self.match)

    @property
    def lemma(self):
        return self.get_lemma(self.match)

    @staticmethod
    def get_func(get_form_string):
        """

        Parameters
        ----------
        get_form_string : str

        Returns
        -------

        """
        nums = list(map(int, get_form_string.split(',')))

        def form_getter(match):
            res = []
            for i in nums:
                res.append(match.group(i))
            return '/'.join(res)
        return form_getter

    def get_type(self, match):
        return '{lemma}/{pos}'.format(lemma=self.get_lemma(match), pos=self.get_pos(match))

    def get_token(self, match, fid, lid):
        return '{lemma}/{pos}/{fid}/{lid}'.format(lemma=self.get_word(match), pos=self.get_pos(match), fid=fid, lid=lid)

    def get_item(self, match, form):
        # form: 'type', 'word', 'lemma'
        if form == 'type':
            return self.get_type(match)
        elif form == 'word':
            return self.get_word(match)
        elif form == 'lemma':
            return self.get_lemma(match)
        else:
            raise ValueError("Unsupported form {}".format(form))


class CorpusFormatter(object):
    connector = '/'  # string for connecting components of a type, token and colloc

    def __init__(self, settings):
        self.match = None
        if 'line-machine' in settings:
            self.line_machine = re.compile(settings['line-machine'])
        elif 'type-line-machine' in settings:
            self._type_line_machine = re.compile(settings['type-line-machine'])
        elif 'colloc-line-machine' in settings:
            self._colloc_line_machine = re.compile(settings['colloc-line-machine'])
        elif 'token-line-machine' in settings:
            self._token_line_machine = re.compile(settings['token-line-machine'])
        else:
            raise ValueError("Please at least provide one line machine!")

        self.bound_mech = settings['boundary-detection-mechanism']
        if self.bound_mech == 'single':
            if 'single-boundary-machine' in settings:
                self._single_bound_machine = re.compile(settings['separator-line-machine'])
            elif 'separator-line-machine' in settings:
                self._single_bound_machine = re.compile(settings['single-boundary-machine'])
            else:
                raise KeyError("You have set single boundary mechanism, "
                               "but you did not set 'single-boundary-machine'!")
        elif self.bound_mech == 'left-right':
            if 'left-boundary-machine' not in settings:
                raise KeyError("You have set left-right boundary mechanism, "
                               "but you did not set 'left-boundary-machine'!")
            if 'right-boundary-machine' not in settings:
                raise KeyError("You have set left-right boundary mechanism, "
                               "but you did not set 'right-boundary-machine'!")
            self._left_bound_machine = re.compile(settings['left-boundary-machine'])
            self._right_bound_machine = re.compile(settings['right-boundary-machine'])
        else:
            raise ValueError('Please set boundary mechanism!')

        columns = settings.get('global-columns', None)
        if columns is None:
            columns = settings.get('line-format', None)
            if columns is None:
                raise ValueError("Please set line format!")
        # 'line-format': e.g. 'word,lemma,pos'
        self.global_columns = columns = columns.split(',')
        col2id = {e: (i+1) for i, e in enumerate(columns)}
        for col, idx in col2id.items():
            self.__dict__['_{column}'.format(column=col)] = idx
        # self._word = col2id.get('word', -1)  # the position (int) in line format
        # self._lemma = col2id.get('lemma', -1)
        # self._pos = col2id.get('pos', -1)

        self.type_format = settings['type']  # e.g. 'lemma/pos'
        type_eles = settings['type'].split(self.connector)
        self._type = [col2id[e] for e in type_eles]  # -> [2, 3]
        self.token_format = settings['token']  # e.g. 'word/pos/fid/lid'
        token_eles = settings['token'].split('/')[:-2]  # the last two parts are always 'fid' and 'lid'
        self._token = [col2id[e] for e in token_eles]  # -> [1, 3]
        if 'colloc' not in settings:  # if not set, use type format as colloc format
            settings['colloc'] = settings['type']
        self.colloc_format = settings['colloc']  # e.g. 'lemma' (means only lemma is used as collocate)
        colloc_eles = settings['colloc'].split('/')
        self._colloc = [col2id[e] for e in colloc_eles]  # -> [2]

        # specify which columns are node attributes or edge attributes
        self.node_attr = settings.get('node-attr', 'word,pos,lemma')
        self.edge_attr = settings.get('edge-attr', None)

        self.settings = deepcopy(settings)  # store settings for possible uses

    def match_line(self, line, form=None):
        if not form:
            return self.line_machine.match(line)
        elif form == 'type':
            return self._type_line_machine.match(line)
        elif form == 'colloc':
            return self._colloc_line_machine.match(line)
        elif form == 'token':
            return self._token_line_machine.match(line)
        else:
            raise NotImplementedError("Only support 'type', 'colloc' and 'token'.")

    def separator_line_machine(self, line):
        return self._single_bound_machine.match(line)

    def left_bound_machine(self, line):
        return self._left_bound_machine.match(line)

    def right_bound_machine(self, line):
        return self._right_bound_machine.match(line)

    def single_bound_machine(self, line):
        return self._single_bound_machine.match(line)

    def get(self, match, column, fid=None, lid=None):
        """Get the content of the corresponding column from a corpus line."""
        if column == 'type':
            return self.get_type(match)
        elif column == 'colloc':
            return self.get_colloc(match)
        elif column == 'token':
            return self.get_token(match, fid, lid)
        else:
            colidx = self.__dict__.get('_{}'.format(column), 0)
            return match.group(colidx)

    def get_word(self, match): # This function and next two copied from old code
        """Get word-form from match object."""
        return match.group(self._word) if self._word != -1 else ''

    def get_pos(self, match):
        """Get part-of-speech tag from match object."""
        return match.group(self._pos) if self._pos != -1 else ''

    def get_lemma(self, match):
        """Get lemma from match object."""
        return match.group(self._lemma) if self._lemma != -1 else ''

    def get_type(self, match):
        """Get type string from match object."""
        return self.connector.join([match.group(i) for i in self._type])

    def get_token(self, match, fid, lid):
        """Get token string from match object."""
        eles = [match.group(i) for i in self._token] + [fid, lid]
        return self.connector.join(eles)

    def get_colloc(self, match):
        """Get token string from match object."""
        return self.connector.join([match.group(i) for i in self._colloc])


class Sentence(object):
    def __init__(self, s):
        self.text = s
        self.content = []
        self.parse()

    def parse(self, s=None):
        """parse sentence text (raw string from corpus file)"""
        if s is None:
            s = self.text
        for line in s.split("\n"):
            if line.startswith('<'):  # skip xml tag lines
                continue
            self.content.append(line)

    def get_content(self):
        return self.content


class PathTemplate(object):
    """Class representing a path template"""
    def __init__(self, nodes, edges):
        self.nodes = deepcopy(nodes)  # a list of str ->
        self.edges = deepcopy(edges)  # a list of str ->
        # self.nodes = map(lambda x: re.compile(x), nodes)
        # self.edges = map(lambda x: re.compile(x), edges)

    @property
    def len(self):
        return len(self.edges)

    def match_node(self, item, index=0):
        m = re.compile(self.nodes[index]).match(item)
        return m

    def match_edge(self, rel, index=0, u=0, v=0):
        m = re.compile(self.edges[index]).match(rel)
        return m

    def __str__(self):
        eles = [self.nodes[0]]
        for i in range(self.len):
            eles.append(self.edges[i])
            eles.append(self.nodes[i+1])
        return ':'.join(eles)


class Path(object):
    """Class storing path matches found in corpus"""
    def __init__(self, template, matches=None):
        self.template = template
        self.matches = matches if matches else []

    @property
    def len(self):
        """size of template
        i.e. '*:NN:amod:VB:*' has a size of one
        """
        return self.template.len

    @property
    def size(self):
        """number of matches"""
        return len(self.matches)

    def add_path(self, match):
        """Add a match

        Parameters
        ----------
        match : iterable
            A list of str
        """
        self.matches.append(match)

    def save(self, filename, encoding='utf-8'):
        pass

    @classmethod
    def load(cls, filename):
        pass
