#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 QLVL <qlvl@kuleuven.be>

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import codecs
import logging
import os

import qlvl
from qlvl.core.terms import Window, CorpusFormatter, ItemNode, TokenNode, TypeNode

logger = logging.getLogger(__name__)


def update_template(filename, settings=None, **kwargs):
    """This is the template of update functions.
    """
    # common part
    formatter = CorpusFormatter(settings)
    input_encoding = settings.get('file-encoding', 'utf-8')

    # get left span and right span
    lspan, rspan = settings['left-span'], settings['right-span']
    # create window
    win = Window(lspan, rspan)

    # process file
    with codecs.open(filename, 'r', input_encoding) as fin:
        lid = 0  # line number (1-based)
        for line in fin:
            lid += 1
            line = line.strip()
            match = formatter.match_line(line)
            if match is None:
                # do something
                pass
            else:
                # if it's a normal line, draws the type from the match
                # do something
                pass
    # deal with the final right window


def update_item_freq(vocab, filename, settings=None):
    """Process lines in file (filename), and add frequencies to vocab.
    !!! this function modifies vocab !!!

    Parameters
    ----------
    vocab : :class:`~qlvl.Vocab`
        The Vocab object to be updated.
    filename : str
        The corpus file name to process
    settings : dict
    """
    formatter = CorpusFormatter(settings)
    input_encoding = settings.get('file-encoding', 'utf-8')

    with codecs.open(filename, 'r', input_encoding) as fin:
        for line in fin:
            line = line.strip()  # in case there is a '\n'
            match = formatter.match_line(line)
            if match is None:
                continue
            # if it's a normal line, draws type from the match
            item_str = formatter.get_type(match)
            vocab.increment(item_str, 1)


def update_col_freq(filename, matrix=None, settings=None):
    """Process lines in file (filename), and add frequencies to matrix.
    !!! this function modifies matrix !!!

    Parameters
    ----------
    filename : str
        Absolute filename of a corpus file
    matrix : 3-tuple
        Includes dict of dict, row item list and column item list.
    settings : dict
    """
    input_encoding = settings.get('file-encoding', 'utf-8')
    lspan, rspan = settings['left-span'], settings['right-span']
    formatter = CorpusFormatter(settings)
    win = Window(lspan, rspan)

    with codecs.open(filename, 'r', input_encoding) as inf:
        lid = 0  # line number
        for line in inf:
            lid += 1
            line = line.strip()
            match = formatter.match_line(line)
            if match is None:  # if it's a separator line
                isseparator = True if formatter.separator_line_machine(line) else False
                if isseparator:
                    # process matches in the right window
                    check_right_window(win, matrix, formatter)
                    win = Window(lspan, rspan)
            else:  # if it's a normal line
                cur = (match, lid)  # record current match into window
                win.update(cur)  # append it to right window
                update_cooccurrence(win, matrix, formatter)
    check_right_window(win, matrix, formatter)


def check_right_window(win, matrix, formatter):
    # handle special cases:
    # [None, ..., None] None [None, .., match, match, .., match]
    if win.node is None:
        i = 0
        while win.node is None and i < win.right_span:
            win.update(None)
            i += 1
        while win.node and i < win.right_span:
            update_cooccurrence(win, matrix, formatter)
            win.update(None)
            i += 1
        return

    # normal cases:
    # [None, .., match, ..] node [match, ..., None] <- None
    while win.node:
        win.update(None)
        update_cooccurrence(win, matrix, formatter)


def update_cooccurrence(win, matrix, formatter):
    """Update co-occurrence frequency matrix with current window.

    Parameters
    ----------
    win : :class:`~qlvl.Window`
        This is a Window object which records current items in span.
        The center item in window is the target word. And it has context
        words of left span and right span stored in two queues.
    matrix : 3-tuple
        Includes dict of dict, row item list and column item list.
    formatter : :class:`~qlvl.CorpusFormatter`
        The getter object for fetching types, tokens and collocs.
    """
    matrix, row_vocab, col_vocab = matrix
    cnode = win.node  # get center node of the window
    if cnode is None:
        return

    match = cnode[0]  # node -> (match, lid) (lid is useful in token level)
    type_ = formatter.get_type(match)

    # TODO: to verify
    # if row_vocab.FILTERPRESENT is True (is set)
    # we should use row_vocab to filter the types we encounter
    # otherwise always found target word (type)
    found = type_ in row_vocab if row_vocab.FILTERPRESENT else True
    if not found:
        return

    lspan, rspan = win.left_span, win.right_span
    for i in range(lspan):
        if win.left[i] is None:  # normally not happen
            # TODO: check never happen
            continue
        match = win.left[i][0]  # (match, lid)
        colloc = formatter.get_colloc(match)

        if not col_vocab.FILTERPRESENT:
            col_vocab.increment(colloc)
            matrix[type_][colloc] += 1
        elif colloc in col_vocab:
            matrix[type_][colloc] += 1
        else:
            # if col_vocab is not empty or colloc not in col_vocab, do nothing
            pass

    for i in range(rspan):
        if win.right[i] is None:  # normally not happen
            # TODO: check never happen
            continue
        match = win.right[i][0]
        colloc = formatter.get_colloc(match)

        if not col_vocab.FILTERPRESENT:
            col_vocab.increment(colloc)
            matrix[type_][colloc] += 1
        elif colloc in col_vocab:
            matrix[type_][colloc] += 1
        else:
            pass


def update_token_nodes(filename, type2tn=None, settings=None):
    """Process lines in file (filename), and match tokens.

    Parameters
    ----------
    filename : str
        Absolute filename of a corpus file
    type2tn : dict
        A dict of key (type string) to value (type node :class:`~qlvl.TypeNode`) mapping..
    settings : dict
    """
    input_encoding = settings.get('file-encoding', 'utf-8')
    lspan, rspan = settings['left-span'], settings['right-span']
    formatter = CorpusFormatter(settings)
    bound_mech = formatter.bound_mech
    win = Window(lspan, rspan)

    fname = os.path.basename(filename).split('.')[0]  # for filename in token
    with codecs.open(filename, 'r', input_encoding) as inf:
        lid = 0
        for line in inf:
            if len(line) <= 0:
                break
            lid += 1  # 1 base line number
            line = line.strip()
            match = formatter.match_line(line)  # use CorpusFormatter object to match a corpus line
            if match is None:   # if it's a separator line
                isseparator = False
                if bound_mech == 'left-right':
                    if formatter.left_bound_machine(line):  # check left boundary
                        isseparator = True
                    if formatter.right_bound_machine(line):  # check right boundary
                        # since meet right boundary, process nodes in the right window
                        process_right_window_tok(win, fname, type2tn, formatter)
                        isseparator = True
                elif bound_mech == 'single':
                    if formatter.single_bound_machine(line):
                        process_right_window_tok(win, fname, type2tn, formatter)
                        isseparator = True
                if isseparator:
                    win = Window(lspan, rspan)
                # else, skip this line
            else:  # normal line
                cur = (match, lid)
                win.update(cur)
                # after we update (append right) a new line to window
                # the center becomes the first item of the right buffer
                # check if the node is in words
                hit_token_for_type(win, fname, type2tn, formatter)
    process_right_window_tok(win, fname, type2tn, formatter)


def process_right_window_tok(win, fname, type2tn, formatter):
    """
    When we meet the end of an article / block,
    we have to check the remaining nodes which are in the right window
    """
    # handle special cases:
    # [None, ..., None] None [None, .., match, match, .., match]
    # TODO: check                                           ^ must be not None???
    if win.node is None and win.right[-1] is not None:
        # update at most right_span times
        while win.node is None:
            win.update(None)
        while win.node:
            hit_token_for_type(win, fname, type2tn, formatter)
            win.update(None)
        return

    while win.node:
        # since we encounter the end of an article
        # we update the window by None
        win.update(None)
        # check if the new center node is in words
        hit_token_for_type(win, fname, type2tn, formatter)


def hit_token_for_type(win, fid, type2tn, formatter):
    cnode = win.node
    if cnode is None:
        return
    match, lid = cnode[0], cnode[1]
    type_ = formatter.get_type(match)
    notfound = type_ not in type2tn
    if notfound:
        return

    # word = formatter.get_word(match)
    # pos = formatter.get_pos(match)
    # lemma = formatter.get_lemma(match)
    # found one token/appearance of a word/type
    # add collocates to the TypeNode object
    tpnode = type2tn[type_]
    # each colloc in window (left and right) is a tuple of (match, lid)
    left_win = [ItemNode(match=colloc[0], formatter=formatter, fid=fid, lid=colloc[1])
                for colloc in list(win.left) if colloc is not None]
    right_win = [ItemNode(match=colloc[0], formatter=formatter, fid=fid, lid=colloc[1])
                 for colloc in list(win.right) if colloc is not None]
    token = TokenNode(fid=fid, lid=lid, match=match, formatter=formatter,
                      lcollocs=left_win, rcollocs=right_win)
    tpnode.append_token(token)
