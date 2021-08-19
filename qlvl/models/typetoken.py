#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: QLVL <qlvl@kuleuven.be>
# Copyright (C) 2018 QLVL KULeuven

"""This module implements the models model.

Other models
============

Usage examples
==============

Initialize a model with e.g.

>>> from qlvl import ItemFreqHandler, ColFreqHandler, TokenHandler
>>> from qlvl.tests.utils import common_texts, get_tmpfile
>>> from qlvl.models import TypeToken
>>>
>>> path = get_tmpfile("models.model")
>>>
>>> model = TypeToken(common_texts, window=(5,5), min_count=1, workers=4)
>>> model.save("models.model")

"""

from __future__ import division

import codecs
import logging
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import qlvl
from qlvl import trange
from qlvl.core.terms import CorpusFormatter, TypeNode, ItemNode, TokenNode
from qlvl.core.vocab import Vocab
from qlvl.core.matrix import TypeTokenMatrix
from qlvl.core.handler import BaseHandler
from qlvl.specutils import mxcalc, mxutils
# from qlvl.specutils.mxutils import transform_spmatrix_to_dict, transform_dict_to_spmatrix, merge_two_matrices
from qlvl.utils import count_values
# from qlvl.models.manager import ItemFreqManager, ColFreqManager, TokenManager, TokenScanner

__all__ = ['ItemFreqHandler', 'ColFreqHandler', 'TokenHandler', 'TypeToken']

logger = logging.getLogger(__name__)
homedir = os.path.expanduser('~')


class ItemFreqHandler(BaseHandler):
    tmpindicator = 'item.freq'

    def __init__(self, settings, workers=0, **kwargs):
        super(ItemFreqHandler, self).__init__(settings, workers=workers, **kwargs)

    def build_item_freq(self, fnames=None):
        """Make a list of all word types that occurred in the corpus
        and write in json format

        Parameters
        ----------
        fnames : str, optional
            Path of file recording corpus file names ('fnames' file of a corpus).
            If this is provided, only the files recorded in this fnames file
            would be processed.
            Else, all files and folders inside the 'corpus-path' of settings
            would be processed.

        Returns
        -------
        vocabulary : :class:`~qlvl.Vocab`
        """
        fnames = self.prepare_fnames(fnames)  # read filenames recorded in 'fnames'
        logger.info("Building item frequency list...")
        vocab = self.process(fnames)
        return Vocab(vocab)

    def process(self, fnames, **kwargs):
        return super(ItemFreqHandler, self).process(fnames, **kwargs)

    def _worker_loop(self, job_queue, res_queue):
        """Worker loop function which gets one by one job from the job queue.
        This function would be executed by many processes.

        Parameters
        ----------
        job_queue : Queue
            A queue of job objects for worker function.
        """
        while True:
            job = job_queue.get()
            if job is None:
                break

            tally = self._do_process_job(job)
            res_queue.put(tally)

        logger.debug("worker exiting")

    def _do_process_job(self, fname, **kwargs):
        """Create an emtpy vocab for each corpus,
        process it by calling `update_item_freq()` function,
        and return the result frequency dict.

        Parameters
        ----------
        fname : str
            One corpus file name

        Returns
        -------
        freq_dict : dict
            The frequency dict of the `fname` file.
        """
        vocab = Vocab()
        # update_item_freq(vocab, fname, self.settings)
        self.update_one_file(fname, vocab)
        return vocab.freq_dict

    def update_one_file(self, filename, data, **kwargs):
        """Process lines in file (filename), and add frequencies to vocab.
        !!! this function modifies vocab !!!

        Parameters
        ----------
        data : :class:`~qlvl.Vocab`
            The Vocab object to be updated.
        filename : str
            The corpus file name to process
        """
        # return super(ItemFreqHandler, self).update_one_file(filename, data)
        with codecs.open(filename, 'r', self.input_encoding) as fin:
            for line in fin:
                line = line.strip()  # in case there is a '\n'
                match = self.formatter.match_line(line)
                if match is None:
                    continue
                # when the current line is a normal (matche) line, draw the type string from the match object
                item_str = self.formatter.get_type(match)
                data.increment(item_str, 1)  # update the vocab

    def _process_results(self, res_queue, n=0):
        """Get all results (frequency dicts) from result queue,
        and merge (add the values) them into one frequency dict.
        """
        vocab = dict()
        for _ in trange(n):
            res = res_queue.get()
            if len(res.keys()) <= 0:
                logger.debug("Empty dict result of one sub-process!")
            for k, v in res.items():
                if k not in vocab:
                    vocab[k] = 0
                vocab[k] += v
        return vocab


class ColFreqHandler(BaseHandler):
    tmpindicator = 'col.freq'
    chunksize = 1000000

    def __init__(self, settings, workers=0, row_vocab=None, col_vocab=None, **kwargs):
        super(ColFreqHandler, self).__init__(settings, workers=workers, **kwargs)
        # if the column vocab is given, then set self.nocolvocab to True
        self.row_vocab = row_vocab if row_vocab else Vocab()
        self.col_vocab = col_vocab if col_vocab else Vocab()

    @property
    def nocolvocab(self):
        return True if len(self.col_vocab) == 0 else False

    def build_col_freq(self, fnames=None, row_vocab=None, col_vocab=None):
        """The function will treat all different word types as possible target or context words.

        Parameters
        ----------
        fnames : str, optional
            Filename of a file which records all (a user wants to process) file names of a corpus.
            Format: corpus_name + settings["fnames-ext"]
        row_vocab : :class:`~qlvl.Vocab`
        col_vocab : :class:`~qlvl.Vocab`

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`
        """
        fnames = self.prepare_fnames(fnames)
        self.row_vocab = row_vocab if row_vocab else self.row_vocab
        self.col_vocab = col_vocab if col_vocab else self.col_vocab
        logger.info("Building collocate frequency matrix...")
        res = self.process(fnames)
        return res

    def process(self, fnames, **kwargs):
        return super(ColFreqHandler, self).process(fnames, **kwargs)

    def _worker_loop_with_colvocab(self, job_queue, res_queue):
        """The worker loop method when the column vocab is provided.
        First we create an empty matrix dict for representing the co-occurrence matrix.
        Then we get one filename from the job_queue and process it by `_do_process_job()`.
        Every time we processed a file, we put an indicator -1 into the res_queue.
        After processed many files, when the number of values (co-occurrence pairs) reaches a pre-set number chunksize,
        the matrix (transformed from the matrix dict and the provided row and col items) would be saved into a tmp file.
        We put this tmp filename into the res_queue.
        Every time we save the matrix, we reset it into empty.
        """
        # 1. prepare variables
        i = 0  # the indicator of the job/res queue
        chunk = 0  # the number of values of the co-occurrence matrix (of these file chunks)
        row_items = self.row_vocab.get_item_list()  # item list of the row vocab
        col_items = self.col_vocab.get_item_list()  # item list of the col vocab
        mtx_dict = defaultdict(lambda: defaultdict(int))  # nested dict for the co-occurrence matrix

        first = True  # used for skipping the first result when showing the progress bar
        while True:
            # 2. this part is the same as the framework `Paralleler._worker_loop()`
            job = job_queue.get()
            if job is None:
                break
            self._do_process_job(job, mtx_dict=mtx_dict)

            # 3. for different tasks, one could add extra data processing part
            chunk = count_values(mtx_dict)
            if chunk > self.chunksize:
                # transform the matrix dict to a TypeTokenMatrix object
                submtx = self.dict2matrix(mtx_dict, row_items, col_items)
                # save the matrix object of current chunk to a temporary file
                tmp_fname = "{}/sub.{}".format(self.subtmpdir, i)
                submtx.save(tmp_fname, pack=False, verbose=False)
                logger.debug("Saved the tmp matrix into {}".format(tmp_fname))
                # put the temporary file name to result queue
                res_queue.put(tmp_fname)

                # reset parameters
                mtx_dict = defaultdict(lambda: defaultdict(int))
                i += 1; chunk = 0
            elif first:
                # do not put the first indicator to the result queue
                first = False
            else:
                # put -1 to the result queue as the indicator for the progress bar
                res_queue.put(-1)

        # for the last chunk
        if chunk > 0:
            submtx = self.dict2matrix(mtx_dict, row_items, col_items)
            tmp_fname = "{}/sub.{}".format(self.subtmpdir, i)
            submtx.save(tmp_fname, pack=False, verbose=False)
            logger.debug("Saved the tmp matrix into {}".format(tmp_fname))
            res_queue.put(tmp_fname)
            del mtx_dict
            i += 1; chunk = 0
        else:
            # as we skip the first one, so for the last chunk, we send an extra indicator
            res_queue.put(-1)

    def _worker_loop_without_colvocab(self, job_queue, res_queue):
        """The worker loop method when the column vocab is not provided."""
        i = 0; chunk = 0
        row_items = self.row_vocab.get_item_list()
        col_items = None  # generate later by the chunk_col_vocab
        self.chunk_col_vocab = Vocab()  # emtpy vocab for current chunk
        mtx_dict = defaultdict(lambda: defaultdict(int))

        first = True
        while True:
            job = job_queue.get()
            if job is None:
                break
            self._do_process_job(job, mtx_dict=mtx_dict)

            chunk = count_values(mtx_dict)
            if chunk > self.chunksize:
                col_items = self.chunk_col_vocab.get_item_list()  # column item list of current chunk
                submtx = self.dict2matrix(mtx_dict, row_items, col_items)
                tmp_fname = "{}/sub.{}".format(self.subtmpdir, i)
                submtx.save(tmp_fname, pack=False, verbose=False)
                logger.debug("Saved the tmp matrix into {}".format(tmp_fname))
                res_queue.put(tmp_fname)

                mtx_dict = defaultdict(lambda: defaultdict(int))
                self.chunk_col_vocab = Vocab()
                i += 1; chunk = 0
            elif first:
                first = False
            else:
                res_queue.put(-1)

        if chunk > 0:
            col_items = self.chunk_col_vocab.get_item_list()
            submtx = self.dict2matrix(mtx_dict, row_items, col_items)
            tmp_fname = "{}/sub.{}".format(self.subtmpdir, i)
            submtx.save(tmp_fname, pack=False, verbose=False)
            logger.debug("Saved the tmp matrix into {}".format(tmp_fname))
            res_queue.put(tmp_fname)

            self.chunk_col_vocab = Vocab()
            i += 1; chunk = 0
        else:
            res_queue.put(-1)

    def _worker_loop(self, job_queue, res_queue):
        """Worker loop function which gets one by one job from the job queue.
        This function would be executed by many threads.

        Parameters
        ----------
        job_queue : Queue
            A queue of job objects for worker function.
        """
        # if the column vocab is not provided
        if self.nocolvocab:
            self._worker_loop_without_colvocab(job_queue, res_queue)
        else:
            self._worker_loop_with_colvocab(job_queue, res_queue)
        logger.debug("worker exiting")

    def _do_process_job(self, fname, mtx_dict=None):
        col_vocab = self.col_vocab if not self.nocolvocab else self.chunk_col_vocab
        matrix = (mtx_dict, self.row_vocab, col_vocab)
        # update_col_freq(fname, matrix=matrix, settings=self.settings)
        self.update_one_file(fname, matrix)

    def update_one_file(self, filename, data, **kwargs):
        return super(ColFreqHandler, self).update_one_file(filename, data)

    def update_one_match(self, matrix, win, lid=0, **kwargs):
        """Update co-occurrence frequency matrix with current window.

        Parameters
        ----------
        matrix : 3-tuple
            Includes dict of dict, row item list and column item list.
        match : :class:`~re.Match`
            A regular expression match object.
        lid : int
            Line number (1-based).
        win : :class:`~qlvl.Window`
            This is a Window object which records current items in span.
            The center item in window is the target word. And it has context
            words of left span and right span stored in two queues.
        """
        # futils.update_cooccurrence(win, matrix, self.formatter)
        mtx_dict, row_vocab, col_vocab = matrix
        cnode = win.node  # get center node of the window
        if cnode is None:
            return

        match = cnode[0]  # node -> (match, lid) (lid is useful in token level)
        type_ = self.formatter.get_type(match)

        # if row_vocab.FILTERPRESENT is True
        # we should use row_vocab to filter the types we encounter
        # otherwise always found target word (type)
        found = type_ in row_vocab if row_vocab.FILTERPRESENT else True
        if not found:  # if not found any type (in the row_vocab), just return
            return

        # if found a type
        lspan, rspan = win.left_span, win.right_span
        for i in range(lspan):
            if win.left[i] is None:  # should not happen
                continue
            # one node in window: (match, lid)
            match, _ = win.left[i]
            colloc = self.formatter.get_colloc(match)

            if not col_vocab.FILTERPRESENT:  # if no given col_vocab
                col_vocab.increment(colloc)  # update the col_vocab
                mtx_dict[type_][colloc] += 1
            elif colloc in col_vocab:  # if the colloc in the given col_vocab
                mtx_dict[type_][colloc] += 1
            else:
                # if col_vocab is not empty or colloc not in col_vocab, do nothing
                pass

        for i in range(rspan):
            if win.right[i] is None:  # should not happen
                continue
            match = win.right[i][0]
            colloc = self.formatter.get_colloc(match)

            if not col_vocab.FILTERPRESENT:
                col_vocab.increment(colloc)
                mtx_dict[type_][colloc] += 1
            elif colloc in col_vocab:
                mtx_dict[type_][colloc] += 1
            else:
                pass

    def process_right_window(self, matrix, win, **kwargs):
        """Process matches in the right window.
        """
        # futils.check_right_window(win, matrix, self.formatter)
        # handle special cases:
        # [None, ..., None] None [None, .., match, match, .., match]
        if win.node is None:
            i = 0
            while win.node is None and i < win.right_span:
                win.update(None)
                i += 1
            while win.node and i < win.right_span:
                self.update_one_match(matrix, win)
                win.update(None)
                i += 1
            return

        # normal cases:
        # [None, .., match, ..] node [match, ..., None] <- None
        while win.node:
            win.update(None)
            self.update_one_match(matrix, win)

    def _process_results(self, res_queue, n=0, **kwargs):
        resmtx = None  # final matrix
        # get results from queue
        for _ in trange(n):
            res = res_queue.get()
            # when data in res_queue is a tmp file name
            if isinstance(res, str):
                mtx = TypeTokenMatrix.load(res, pack=False)
                if not resmtx:
                    resmtx = mtx
                else:
                    resmtx = mxutils.merge_two_matrices(resmtx, mtx)
                # remove temporary files (*.meta, *.npz)
                try:
                    os.remove('{}.{}'.format(res, 'meta'))
                    os.remove('{}.{}'.format(res, 'npz'))
                except Exception as err:
                    logger.exception("Cannot remove *.meta or *.npz tmp files.\n{}".format(err))
            # else: the indicator -1, do nothing

        return resmtx

    @staticmethod
    def dict2matrix(mtx_dict, row_items, col_items, classname=TypeTokenMatrix):
        spmtx = mxutils.transform_dict_to_spmatrix(mtx_dict, row_items, col_items)
        return classname(spmtx, row_items, col_items)


class TokenHandler(BaseHandler):
    """Handler Class for retrieving tokens"""
    tmpindicator = 'tok.app'
    chunksize = 1000000

    def __init__(self, queries, settings=None, workers=0, row_vocab=None, col_vocab=None, **kwargs):
        """
        Parameters
        ----------
        queries : iterable or :class:`~qlvl.Vocab`
            Target types (queries) vocabulary. Must provide this.
            Only retrieve tokens of these types.
        col_vocab : :class:`~qlvl.Vocab`
            Context features vocabulary.
            If a non-empty vocabulary is passed, only context features in this vocab
            should be processed.
            Otherwise all possible contexts should be processed.

        Notes
        -----
        The first argument is the queries, not the settings (like ItemFreqHandler and ColFreqHandler)
        """
        super(TokenHandler, self).__init__(settings, workers=workers, **kwargs)
        # if the column vocab is given, then set self.nocolvocab to True
        self.col_vocab = col_vocab if col_vocab else Vocab()
        self.nocolvocab = True if len(self.col_vocab) == 0 else False

        self.formatter = CorpusFormatter(self.settings)
        self.type2toks = {w: TypeNode(type_str=w, type_fmt=self.formatter.type_format)
                          for w in queries.keys()}

    def retrieve_tokens(self, fnames=None):
        """Scan/Retrieve tokens from corpus files.

        Parameters
        ----------
        fnames : str, optional
            Filename of a file which records all (a user wants to process) file names of a corpus.
            Format: corpus_name + settings["fnames-ext"]

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`
        """
        fnames = self.prepare_fnames(fnames)
        logger.info("Scanning tokens of queries in corpus...")
        res = self.process(fnames)
        return mxutils.transform_nodes_to_matrix(res)

    def process(self, fnames, **kwargs):
        return super(TokenHandler, self).process(fnames, **kwargs)

    def _worker_loop(self, job_queue, res_queue):
        """Worker loop function which gets one by one job from the job queue.
        This function would be executed by many threads.

        Parameters
        ----------
        job_queue : Queue
            A queue of job objects for worker function.
        """
        first = True
        while True:
            job = job_queue.get()
            if job is None:
                break
            self._do_process_job(job)

            if first:
                first = False
            else:
                res_queue.put(-1)

        # tmp_fname = "{}/sub.0".format(self.subtmpdir)
        res_queue.put(self.type2toks)
        logger.debug("worker exiting")

    def _do_process_job(self, fname, **kwargs):
        # update_token_nodes(fname, type2tn=self.type2toks, settings=self.settings)
        self.update_one_file(fname, self.type2toks)

    def update_one_file(self, filename, data, **kwargs):
        return super(TokenHandler, self).update_one_file(filename, data)

    def update_one_match(self, type2toks, win, fid='fname', **kwargs):
        """

        Parameters
        ----------
        type2toks : dict
            A dict mapping from a type string to the token nodes of it.
        win
        fid : str
            The file id in a token string.
        kwargs

        Returns
        -------

        """
        # after we update (append right) a new line to window
        # the center becomes the first item of the right buffer
        # check if the node is in words
        # futils.hit_token_for_type(win, fname, type2toks, self.formatter)
        cnode = win.node
        if cnode is None:
            return
        match, lid = cnode[0], cnode[1]
        type_ = self.formatter.get_type(match)
        notfound = type_ not in type2toks
        if notfound:
            return

        # word = formatter.get_word(match)
        # pos = formatter.get_pos(match)
        # lemma = formatter.get_lemma(match)
        # found one token/appearance of a word/type
        # add collocates to the TypeNode object
        tpnode = type2toks[type_]
        # each colloc in window (left and right) is a tuple of (match, lid)
        left_win = [ItemNode(match=colloc[0], formatter=self.formatter, fid=fid, lid=colloc[1])
                    for colloc in list(win.left) if colloc is not None]
        right_win = [ItemNode(match=colloc[0], formatter=self.formatter, fid=fid, lid=colloc[1])
                     for colloc in list(win.right) if colloc is not None]
        if not self.nocolvocab:
            left_win = [x for x in left_win if x.to_type() in self.col_vocab.get_item_list()]
            right_win = [x for x in right_win if x.to_type() in self.col_vocab.get_item_list()]
        token = TokenNode(fid=fid, lid=lid, match=match, formatter=self.formatter,
                          lcollocs=left_win, rcollocs=right_win)
        tpnode.append_token(token)

    def process_right_window(self, type2toks, win, fid='fname', **kwargs):
        """When we meet the end of an article / block,
        we have to check the remaining nodes which are in the right window
        """
        # futils.process_right_window_tok(win, fname, type2toks, self.formatter)

        # handle special cases:
        # [None, ..., None] None [None, .., match, match, .., match]
        # TODO: check                                           ^ must be not None???
        if win.node is None and win.right[-1] is not None:
            # update at most right_span times
            while win.node is None:
                win.update(None)
            while win.node:
                self.update_one_match(type2toks, win, fid=fid)
                win.update(None)
            return

        while win.node:
            # since we encounter the end of an article
            # we update the window by None
            win.update(None)
            # check if the new center node is in words
            self.update_one_match(type2toks, win, fid=fid)

    def _process_results(self, res_queue, n=0, **kwargs):
        type2toks = defaultdict(list)

        for _ in trange(n):
            res = res_queue.get()
            # when data in res_queue is a `type2toks` dict
            if isinstance(res, dict):
                for k, v in res.items():
                    type2toks[k].append(v)
        type2toks = {k: TypeNode.merge(v) for k, v in type2toks.items()}
        return type2toks


class TypeVectorizer(object):
    """Convert a collection of text documents (corpus files) to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer that does
    some kind of feature selection then the number of features will be equal to the vocabulary
    size found by analyzing the data.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : string {'filename'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    See also
    --------
    DepRelVectorizer

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """
    def __init__(self, settings=None, corpus_name=None,
                 row_vocab=None, col_vocab=None,
                 row_max_df=1.0, row_min_df=1,
                 col_max_df=1.0, col_min_df=1,
                 dtype=np.int64):
        self.settings = deepcopy(settings)
        self.corpus_path = self.settings.get('corpus-path', None)
        self.corpus_name = corpus_name if not corpus_name else self.corpus_path.split(os.sep)[-1]
        self.input_encoding = settings.get('file-encoding', 'utf-8')
        self.output_encoding = settings.get('output-encoding', 'utf-8')
        self.row_vocab = row_vocab
        self.row_max_df = row_max_df
        self.row_min_df = row_min_df
        self.col_vocab = col_vocab
        self.col_max_df = col_max_df
        self.col_min_df = col_min_df
        if row_max_df < 0 or row_min_df < 0 or col_max_df < 0 or col_min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.dtype = dtype

    def get_settings(self):
        return deepcopy(self.settings)

    def build_vocab(self, fnames=None):
        """A utility method for build a vocabulary.

        Parameters
        ----------
        fnames : str or iterable, optional
            Filename of a file which records all (a user wants to process) file names of a corpus.
            Format: corpus_name + settings["fnames-ext"]

        Returns
        -------
        vocab : :class:`~qlvl.core.vocab.Vocab`
        """
        ifhan = ItemFreqHandler(settings=self.settings)
        vocab = ifhan.build_item_freq(fnames=fnames)
        return vocab

    def fit(self, fnames=None):
        """Build a co-occurrence frequency matrix of tokens in the raw documents.

        Parameters
        ----------
        fnames : str or iterable, optional
            Filename of a file which records all (a user wants to process) file names of a corpus.
            Format: corpus_name + settings["fnames-ext"]

        Returns
        -------
        freqmx : :class:`~qlvl.core.matrix.TypeTokenMatrix`
            Co-occurrence frequency matrix.
        """
        cfhan = ColFreqHandler(self.settings)
        if self.row_vocab is None:
            logger.info("Building vocabulary since no vocabulary provided!")
            ifhan = ItemFreqHandler(settings=self.settings)
            self.row_vocab = ifhan.build_item_freq(fnames=fnames)
        if self.col_vocab is None:
            self.col_vocab = self.row_vocab.deepcopy()
        self.freqmx = cfhan.build_col_freq(fnames=fnames, row_vocab=self.row_vocab, col_vocab=self.col_vocab)
        # TODO: filter matrix by max_df and min_df

        return self.freqmx

    def vectorize(self, meas='ppmi'):
        """Compute association measures matrix.

        Parameters
        ----------
        meas : str
            Could be: 'pmi', 'ppmi', 'lik', 'chisq', 'zscore', 'dice', 'deltap', 'logratio'.

        Returns
        -------
        vectors : :class:`qlvl.core.matrix.TypeTokenMatrix`
        """
        self.vectors = mxcalc.compute_association(self.freqmx, meas=meas)
        return self.vectors


class TokenVectorizer(object):
    def __init__(self, concepts, settings=None, corpus_name=None, dtype=np.int8):
        """

        Parameters
        ----------
        concepts : dict (str -> list of str)
            Concepts dict maps concepts (string) to the corresponding variants (lists of items)
        settings
        corpus_name
        dtype
        """
        self.settings = deepcopy(settings)
        self.corpus_path = self.settings.get('corpus-path', None)
        self.corpus_name = corpus_name if not corpus_name else self.corpus_path.split(os.sep)[-1]
        self.input_encoding = settings.get('file-encoding', 'utf-8')
        self.output_encoding = settings.get('output-encoding', 'utf-8')
        self.concept_queries = concepts

    def get_settings(self):
        return deepcopy(self.settings)

    def retrieve_tokens(self, fnames=None):
        query_dict = {}
        for concept, variants in self.concept_queries.items():
            queries = {l: 0 for l in variants}
            query_dict.update(queries)

        tokhan = TokenHandler(Vocab(query_dict), settings=self.settings)
        typenodes = tokhan.retrieve_tokens(fnames=fnames)

        self.concepts = dict()
        for concept, variants in self.concept_queries.items():
            self.concepts[concept] = []
            for v in variants:
                self.concepts[concept].append(typenodes[v])
        return self.concepts

    def generate_vectors(self, freqMTX, freq_cutoff=0, ppmi_cutoff=0.0, llr_cutoff=0.0):
        concept2vectors = dict()
        for concept, variants in self.concepts.items():
            variant_list = [str(v) for v in variants]
            cep_freqMTX = freqMTX.submatrix(row=variant_list)
            cutoff_freqMTX = cep_freqMTX.multiply(cep_freqMTX > freq_cutoff)
            cutoff_freqMTX = cutoff_freqMTX.drop_empty(axis=1)

            node_vocab = Vocab(freqMTX.sum(axis=1))
            colloc_vocab = Vocab(freqMTX.sum(axis=0))
            cep_ppmiMTX = mxcalc.compute_association(cutoff_freqMTX, nfreq=node_vocab, cfreq=colloc_vocab, meas='ppmi')
            cep_llrMTX = mxcalc.compute_association(cutoff_freqMTX, nfreq=node_vocab, cfreq=colloc_vocab, meas='lik')
            cutoff_ppmiMTX = cep_ppmiMTX.multiply(cep_ppmiMTX > ppmi_cutoff)
            cutoff_ppmiMTX = cutoff_ppmiMTX.multiply(cep_llrMTX > llr_cutoff)
            cutoff_ppmiMTX = cutoff_ppmiMTX.drop_empty(axis=1)

            # eliminate FOCs from Boolean matrix that are not found in the context of the tokens
            # if done, final_ppmiMTX should do submatrix() as well
            cep_tokmx = mxutils.transform_nodes_to_matrix(variants)
            intersect_collocs = list(set(cutoff_ppmiMTX.col_items).intersection(set(cep_tokmx.col_items)))
            final_cep_ppmiMTX = cutoff_ppmiMTX.submatrix(col=intersect_collocs)
            final_cep_tokmx = cep_tokmx.submatrix(col=intersect_collocs)

            # compute token-by-FOCs ppmi matrix
            tokweights = mxcalc.compute_token_weights(final_cep_tokmx, final_cep_ppmiMTX)
            final_tokweights = tokweights.drop_empty(axis=0)

            # subset the full collocate frequency matrix based on candidates FOCs and SOCs
            colloc_freqMTX = freqMTX.submatrix(row=intersect_collocs)
            soc_ppmiMTX = mxcalc.compute_association(colloc_freqMTX, nfreq=node_vocab, cfreq=colloc_vocab, meas='ppmi')
            cep_tokvecs = mxcalc.compute_token_vectors(final_tokweights, soc_ppmiMTX)
            concept2vectors[concept] = cep_tokvecs

        return concept2vectors


class TypeToken(object):
    """Train, use and evaluate models model.

    Attributes
    ----------
    fv : collocate frequencies matrix
    tv : token vectors

    vocabulary :
        This object represents the (all items) vocabulary of a corpus.
    row_vocab :
        Row vocabulary.
    col_vocab :
        Column vocabulary.

    """
    def __init__(self, settings=None, corpus_name=None):
        self.settings = deepcopy(settings)
        self.corpus_path = self.settings.get('corpus-path', None)
        self.corpus_name = corpus_name if not corpus_name else self.corpus_path.split(os.sep)[-1]
        self.output_path = self.settings.get('output-path', homedir)
        self.input_encoding = settings['file-encoding']
        self.output_encoding = settings['output-encoding']
        self.vocab = None
        self.freqmx = None
        self.measmx = None
        self.simmx = None
        self.distmx = None
        self.tcmx = None
        self.token_weights = None
        self.token_vectors = None

    def get_settings(self):
        return deepcopy(self.settings)

    def get_vocab(self):
        return self.vocab.deepcopy()

    def build_frequency_list(self, fnames=None, multicore=True, prog_bar=True):
        """Alias method of build_vocab()."""
        return self.build_vocab(fnames=fnames, multicore=multicore, prog_bar=prog_bar)

    def build_vocab(self, fnames=None, multicore=True, prog_bar=True):
        """A caller method of TypeToken model.
        It calls the same method `ItemFreqHandler.build_vocab`.

        Parameters
        ----------
        fnames : str
            Path of file recording corpus file names ('fnames' file of a corpus).
            If this is provided, only the files recorded in this fnames file
            would be processed.
            Else, all files and folders inside the 'corpus-path' of settings
            would be processed.
        multicore : bool
            Use multicore processing or not.
        prog_bar : bool
            Show progress bar or not.

        Returns
        -------
        :class:`~qlvl.Vocab`

        See Also
        --------
        build_vocab : :class:`~qlvl.ItemFreqHandler.build_vocab`
        """
        ifman = ItemFreqManager(self.settings)
        self.vocab = ifman.build_item_freq(fnames=fnames, multicore=multicore, prog_bar=prog_bar)
        return self.vocab.deepcopy()

    def build_col_freq(self, fnames=None, row_vocab=None, col_vocab=None, multicore=True, prog_bar=True):
        cfman = ColFreqManager(self.settings)
        row_vocab = row_vocab if row_vocab else self.vocab
        col_vocab = col_vocab if col_vocab else Vocab()
        self.freqmx = cfman.build_col_freq(fnames=fnames, row_vocab=row_vocab, col_vocab=col_vocab,
                                           multicore=multicore, prog_bar=prog_bar)
        return self.freqmx.deepcopy()

    def compute_association(self, freqmx=None, meas='ppmi'):
        """Compute association measures matrix.

        Parameters
        ----------
        freqmx
        meas : str
            Could be: 'pmi', 'ppmi', 'lik', 'chisq', 'zscore', 'dice', 'deltap', 'logratio'.

        Returns
        -------

        """
        freqmx = freqmx if freqmx else self.freqmx
        self.measmx = mxcalc.compute_association(freqmx, meas=meas)
        return self.measmx

    def compute_similarity(self, measmx=None, metric='cosine', rank=False, axis=0):
        """Compute similarity matrix.

        Parameters
        ----------
        measmx : :class:`~qlvl.TypeTokenMatrix`
        metric : str
            Could be: 'cos', 'rank'

        Returns
        -------

        """
        measmx = measmx if measmx else self.measmx
        if metric == 'cos' or metric == 'cosine':
            self.simmx = mxcalc.compute_cosine(measmx, axis=axis)
        else:
            raise NotImplementedError("Not implement this metric: {}".format(metric))
        if rank:
            self.rankmx = mxcalc.compute_simrank(self.simmx)
        return self.simmx

    def compute_distance(self, measmx=None, metric='cosine'):
        """Compute distance matrix.

        Parameters
        ----------
        measmx : :class:`~qlvl.TypeTokenMatrix`
        metric : str
            Could be: 'cos', 'rank'

        Returns
        -------

        """
        measmx = measmx if measmx else self.measmx
        self.simmx = mxcalc.compute_distance(measmx, metric=metric)
        return self.simmx

    def compute_simrank(self, simmx=None, distance=False, reverse=False):
        """Compute similarity rank matrix.

        Parameters
        ----------
        simmx
        distance
        reverse

        Returns
        -------

        """
        # if distance:  # the input matrix is a distance matrix
        self.rankmx = mxcalc.compute_simrank(simmx, reverse=reverse)
        return self.rankmx.deepcopy()

    def select_types(self, type_list):
        """

        Parameters
        ----------
        type_list : list of str

        Returns
        -------

        """
        if not isinstance(type_list, list):
            raise ValueError("Please pass a list of types!")
        if not self.vocab:
            raise ValueError("Model has no vocabulary, please set one (use TypeToken.set_vocab())!")

        return self.vocab.subvocab(type_list)

    def retrieve_tokens(self, fnames=None, queries=None, row_vocab=None, col_vocab=None, multicore=True, prog_bar=True):
        if queries:
            if row_vocab:
                raise ValueError("Please specify only one to use: queries or row_vocab!")
            row_vocab = self.select_types(queries)
        elif not row_vocab:
            raise ValueError("Please pass queries or row_vocab!")
        tcman = TokenManager(self.settings)
        self.tcmx = tcman.retrieve_tokens(fnames=fnames, row_vocab=row_vocab, col_vocab=col_vocab, multicore=multicore, prog_bar=prog_bar)
        return self.tcmx.deepcopy()

    def build_token_weights(self, tcPositionMTX=None, twMTX=None):
        """Build token-context weight matrix.

        Parameters
        ----------
        tcPositionMTX : token context position matrix
                                   target words
                                 ---------------
                                |               |
                         tokens |               |
                                |               |
                                 ---------------
        twMTX : type weight matrix, ex. 'pmi' (transposed)
                                   target words
                                 ---------------
                        context |      ...      |
                       features | ...   x   ... |
                        (types) |      ...      |
                                 ---------------

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`

        """
        if tcPositionMTX is None:
            tcPositionMTX = self.tcmx
        if twMTX is None:
            raise ValueError("Please input a valid twMTX!")
        self.token_weights = build_token_weights(tcPositionMTX, twMTX)
        return self.token_weights.deepcopy()

    def build_token_vectors(self, tcWeightMTX=None, soccMTX=None, operation='addition'):
        """Build token vectors.

        Parameters
        ----------
        soccMTX : :class:`~qlvl.TypeTokenMatrix`
            Second order collocate matrix.
        tcWeightMTX : :class:`~qlvl.TypeTokenMatrix`
            Token-Context weight matrix.
        operation : str
            'addition', 'multiplication'

        Returns
        -------

        """
        tcWeightMTX = tcWeightMTX if tcWeightMTX is not None else self.token_weights
        if tcWeightMTX is None:
            raise ValueError("Please input a valid token weight matrix!")
        if soccMTX is None:
            raise ValueError("Please input a valid second order matrix!")
        self.token_vectors = build_token_vectors(tcWeightMTX, soccMTX, operation=operation)
        return self.token_vectors.deepcopy()

    def read_queries(self, queries):
        default_key = self.settings['wqueries-default-key']
        if not isinstance(queries, list):
            # a list of word/type pairs: (specified word, neutral word)
            self.type_pairs = [(get_word_str(w, specific=True, corpus_name=self.corpus_name, def_key=default_key),
                                get_word_str(w, specific=False, def_key=default_key))
                               for w in queries]
        pass

    def fetch_tokens(self, queries, fnames=None, multicore=True, prog_bar=True):
        subvocab = self.vocab.subvocab(queries)
        tr = TokenScanner(self.settings, queries)
        tr.set_raw_vocab(subvocab)
        tpnodes = tr.retrieve_tokens(fnames=fnames, multicore=multicore, prog_bar=prog_bar)
        self.type_nodes = tpnodes
        return tpnodes

    def sample_tokens(self, n=300, method='random'):
        type_nodes = dict()
        for tp, node in self.type_nodes.items():
            new_node = node.sample(n=n, method=method)
            type_nodes[tp] = new_node
        return type_nodes

    def make_token_colloc(self, type_nodes=None, colloc_vocab=None, prog_bar=True):
        type_nodes = type_nodes if type_nodes else self.type_nodes
        type_mtxs = dict()
        for tp, tn in type_nodes.items():
            # token-context position matrix of a type
            tpmx = read_token_colloc(tn, colloc_vocab=colloc_vocab, settings=self.settings)
            token_strs = sorted(tpmx.keys())
            if colloc_vocab:
                colloc_strs = colloc_vocab.get_item_list()
            else:
                colloc_strs = set()
                for tok_str, node in tpmx.items():
                    for colloc, _ in node.items():
                        colloc_strs.add(colloc)
                colloc_strs = sorted(colloc_strs)
            spmatrix = mxutils.transform_dict_to_spmatrix(tpmx, token_strs, colloc_strs)
            tpMTX = TypeTokenMatrix(spmatrix, token_strs, colloc_strs)
            type_mtxs[tp] = tpMTX

        return type_mtxs


def read_token_colloc(type_node, colloc_vocab=None, settings=None):
    """

    Parameters
    ----------
    type_node : :class:`~qlvl.TypeNode`
    colloc_vocab : :class:`~qlvl.Vocab`
        collocate vocabulary
    settings : dict

    Returns
    -------

    """
    lspan, rspan = settings['left-span'], settings['right-span']
    colloc_fmt = settings.get('colloc', 'lemma/pos')
    matrix = defaultdict(lambda: defaultdict(int))
    for tok in type_node.tokens:
        tok_str = str(tok)
        for i in range(min(lspan, tok.lspan)):
            position = -(i + 1)
            colloc = tok.lcollocs[position]  # -> ItemNode
            colloc_str = colloc.to_colloc(colloc_fmt=colloc_fmt)
            if not colloc_vocab or colloc_str in colloc_vocab:
                matrix[tok_str][colloc_str] = position
        for i in range(min(rspan, tok.rspan)):
            position = i + 1
            colloc = tok.rcollocs[i]
            colloc_str = colloc.to_colloc(colloc_fmt=colloc_fmt)
            if not colloc_vocab or colloc_str in colloc_vocab:
                matrix[tok_str][colloc_str] = position
    return matrix


# old implementation of this function
def build_tc_weight_matrix(tcPositionMTX, twMTX):
    tok_context_mtx = mxutils.transform_spmatrix_to_dict(tcPositionMTX.matrix, tcPositionMTX.row_items, tcPositionMTX.col_items)
    type_weight_mtx = mxutils.transform_spmatrix_to_dict(twMTX.matrix, twMTX.row_items, twMTX.col_items)

    tok_weight_dict = defaultdict(lambda: defaultdict(float))
    missing_types = []
    # split tokens into groups of each type
    tokens = tcPositionMTX.row_items
    type2token = dict()
    for tok in tokens:
        type_ = '/'.join(tok.split('/')[:-2])
        if type_ not in type_weight_mtx:
            missing_types.append(type_)
            continue
        if type_ not in type2token:
            type2token[type_] = []
        type2token[type_].append(tok)

    for type_, toks in type2token.items():
        tw_row = type_weight_mtx[type_]
        for tok in toks:
            tc_row = tok_context_mtx[tok]
            tc_feats = tc_row.keys()
            for feat in tc_feats:
                if feat not in tw_row:
                    continue
                val = tw_row[feat]
                tok_weight_dict[tok][feat] = val

    tc_weight_mtx = mxutils.transform_dict_to_spmatrix(tok_weight_dict, tcPositionMTX.row_items, twMTX.col_items)
    return TypeTokenMatrix(tc_weight_mtx, tcPositionMTX.row_items, tcPositionMTX.col_items)
