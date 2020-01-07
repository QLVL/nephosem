try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import codecs
import logging
import os
from collections import defaultdict
from copy import deepcopy
from multiprocessing import cpu_count, Pool

from qlvl import progbar
from qlvl.bak.futils import check_right_window, process_right_window_tok, hit_token_for_type, update_cooccurrence
from qlvl.core.matrix import TypeTokenMatrix
from qlvl.core.terms import CorpusFormatter, TypeNode, Window
from qlvl.core.vocab import Vocab
from qlvl.specutils.mxutils import transform_dict_to_spmatrix, merge_matrices
from qlvl.utils import timeit, read_fnames, read_fnames_of_corpus, make_dir, clean_dir, pickle, unpickle

logger = logging.getLogger(__name__)
homedir = os.path.expanduser('~')


class BaseManager(object):
    """This is a base class of all handler classes.

    Contains framework for multicore method. The purpose of this class is to provide a
    reference interface for concrete handler implementations. At the same time,
    functionality that we expect to be common for those implementations is provided
    here to avoid code duplication.

    A typical procedure of processing a corpus would be

    Attributes
    ----------
    settings : dict
    corpus_path : str
    output_path : str
    encoding : str
        Default 'utf-8'
    input_encoding : str
        File encoding of input corpus files.
        Default 'utf-8'
    output_encoding : str
        File encoding of output files.
        Could be different with input_encoding.
        e.g. :
        If input_encoding is 'latin-1', but we don't want to use it for output files.
        We could use 'utf-8' for output files.
        Default 'utf-8'.

    Notes
    -----
    A subclass should initialize the following attributes:

    * self.settings - settings dict

    """

    def __init__(self, settings, workers=0):
        self.settings = deepcopy(settings)
        self.corpus_path = self.settings.get('corpus-path', None)
        self.output_path = self.settings.get('output-path', homedir)
        self.encoding = self.settings.get('file-encoding', 'utf-8')
        # TODO: choose from input_encoding, file_encoding or encoding
        self.input_encoding = self.encoding
        self.output_encoding = self.settings.get('outfile-encoding', self.encoding)
        self.workers = int(workers) if 0 < int(workers) < cpu_count() else cpu_count() - 1

    def do_job_single(self, fnames, **kwargs):
        """Do the job using a single cpu core."""
        raise NotImplementedError

    def prepare_fnames(self, fnames=None):
        """Prepare corpus file names based on the fnames file path or the corpus path.
        If a valid `fnames` is passed, read all file names recorded inside this file.
        If not, use the corpus directory `self.corpus_path` and read all file names inside this folder.

        Parameters
        ----------
        fnames : str or list, optional
            If str, then it is the filename of a file which records all (a user wants to process) file names of a corpus.
            If list, then it contains a list of filenames.

        """
        if fnames is None:
            fnames = read_fnames_of_corpus(self.corpus_path)
        else:
            if isinstance(fnames, list):
                pass  # do nothing, already a list of (string) file names
            elif isinstance(fnames, str):
                fnames = read_fnames(fnames, self.encoding)
            else:
                raise ValueError("Not support other types of 'fnames' (only str or a list of strings)!")
        return fnames

    def prepare_data_group(self, data):
        """Split file names into many groups for sub-processes.

        Parameters
        ----------
        data : iterable
            Normally data is a list of corpus file names.

        Returns
        -------
        data groups : iterable
            A list of data groups.
        """
        num_groups = self.workers  # the number of sub-processes / data groups
        num_data = len(data)
        data_group = [[] for _ in range(num_groups)]
        for i in range(num_data):  # loop over all data and add one to a group
            idx = i % num_groups
            data_group[idx].append(data[i])
        return data_group

    def do_job_multicore(self, fnames, job_func, **kwargs):
        """Generate sub-processes and run based on the given `job_func`.
        Common implementation of multicore method of different handlers while using different job function respectively.

        Parameters
        ----------
        fnames : iterable
            A list of file names prepared.
        job_func : function
            Target function to run for sub-processes.
        """
        data_group = self.prepare_data_group(fnames)
        pool = Pool(processes=self.workers)  # -> a pool of `self.workers` sub-processes
        for i in range(self.workers):
            args = (data_group[i],)
            # use data groups as `args` and directly use passed `kwargs`
            pool.apply_async(job_func, args=args, kwds=kwargs)
        pool.close()
        pool.join()

        result = self.merge_results()
        return result

    def merge_results(self):
        """Merge results of sub-processes and return."""
        raise NotImplementedError


class ItemFreqManager(BaseManager):
    """Handler Class for counting item frequency list (dict)

    Some important internal attributes are the following:

    Attributes
    ----------
    tmpdir : str
        Temporary directory. This directory will be automatically created during a multicore procedure.

    """
    def __init__(self, settings):
        super(ItemFreqManager, self).__init__(settings)
        # TODO: use which directory as tmp folder
        # current: /output_path/tmp/item.freq/
        self.tmpdir = os.path.join(self.output_path, 'tmp', 'item.freq')
        make_dir(self.tmpdir)   # create tmpdir if not exists
        clean_dir(self.tmpdir)  # clean tmpdir if exists

    def make_item_freq(self, fnames=None, multicore=True, prog_bar=True):
        """Alias of build_item_freq() method."""
        return self.build_item_freq(fnames=fnames, multicore=multicore, prog_bar=prog_bar)

    @timeit
    def build_item_freq(self, fnames=None, multicore=True, prog_bar=True):
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
        multicore : bool
            Use multicore processing or not.
        prog_bar : bool
            Show the progress bar or not.

        Returns
        -------
        vocabulary : :class:`~qlvl.Vocab`
        """
        fnames = self.prepare_fnames(fnames)  # read filenames recorded in 'fnames'
        logger.info("Building item frequency list...")
        if multicore:
            vocab = self.do_job_multicore(
                fnames, update_item_freq_caller,
                tmpdir=self.tmpdir, settings=self.settings,
                prog_bar=prog_bar)
        else:
            vocab = self.do_job_single(fnames, prog_bar=prog_bar)
        return vocab

    def do_job_single(self, fnames, prog_bar=True):
        """Method doing job for handler class.

        Parameters
        ----------
        fnames : iterable
            A list of filenames
        job_func :
            Target function to run
        prog_bar

        Returns
        -------
        :class:`~qlvl.Vocab`
        """
        pfnames = progbar(fnames, unit='file', desc='  corpus') if prog_bar else fnames

        vocab = Vocab(dict(), encoding=self.output_encoding)
        for fname in pfnames:
            args = (vocab, fname,)
            kwargs = {'settings': self.settings}
            try:
                update_item_freq(*args, **kwargs)
            except Exception as e:
                logger.exception("{} error:\n{}".format(fname, e))

        return vocab.copy()
        # vocab = Vocab(vocab.get_dict(), encoding=self.output_encoding)
        # return vocab

    def merge_results(self):
        """Merge sub-process results (Vocab) in tmp folder."""
        freq_dict = dict()  # final result dict
        for fname in os.listdir(self.tmpdir):
            fname = os.path.join(self.tmpdir, fname)
            tmp_vocab = Vocab.load(fname, self.output_encoding)  # load a vocab of a sub-process
            tmp_dict = tmp_vocab.get_dict()
            # update results
            for k, v in tmp_dict.items():
                if k in freq_dict:
                    freq_dict[k] += v
                else:
                    freq_dict[k] = v
            try:
                os.remove(fname)
            except Exception as err:
                logger.warning("Cannot remove tmp file {}\n{}".format(fname, err))
        try:
            os.rmdir(self.tmpdir)
        except Exception as err:
            logger.warning("Cannot remove tmp folder {}\n{}".format(self.tmpdir, err))
        return Vocab(freq_dict, encoding=self.output_encoding)


def update_item_freq_caller(fnames, tmpdir=None, settings=None, prog_bar=True):
    """Caller function for multicore method.
    Calls update_item_freq().

    Parameters
    ----------
    fnames : iterable
        A list of filenames
    tmpdir : str
        Temporary folder
    settings : dict
    prog_bar :
    """
    if not tmpdir:  # if tmpdir is not set
        tmpdir = os.path.join(homedir, 'tmp')  # use "/home/user/tmp"
    if not settings:
        raise ValueError("Please pass a valid settings!")
    output_encoding = settings.get('output-encoding', 'utf-8')

    pid = os.getpid()
    # if print something in subprocess, then the jupyter notebook will show progress bar (???)
    logger.info("Starting subprocess {}...".format(pid))
    pfnames = progbar(fnames, unit='file', desc='  corpus') if prog_bar else fnames

    vocab = Vocab(dict(), encoding=output_encoding)
    for fname in pfnames:
        args = (vocab, fname,)
        kwargs = {'settings': settings}
        try:
            update_item_freq(*args, **kwargs)
        except Exception as e:
            logger.error("{} error:\n{}".format(fname, e))

    # save vocab to tmp file, instead of return vocab in single-core function
    tmp_proc_fname = os.path.join(tmpdir, '{}.vocab'.format(pid))
    try:
        vocab.save(tmp_proc_fname, verbose=False)
    except Exception as err:
        logger.exception(err)


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
            # if it's a normal line, draws type from match
            item_str = formatter.get_type(match)
            vocab.increment(item_str, 1)


class ColFreqManager(BaseManager):
    """Handler Class for counting co-occurrence frequency matrix"""
    def __init__(self, settings):
        super(ColFreqManager, self).__init__(settings)
        self.tmpdir = os.path.join(self.output_path, 'tmp', 'col.freq')
        make_dir(self.tmpdir)   # create tmpdir if not exists
        clean_dir(self.tmpdir)  # clean tmpdir if exists

    def make_col_freq(self, fnames=None, row_vocab=None, col_vocab=None, multicore=True, prog_bar=True):
        """Alias of build_col_freq() method."""
        return self.build_col_freq(fnames=fnames, row_vocab=row_vocab, col_vocab=col_vocab, multicore=multicore, prog_bar=prog_bar)

    @timeit
    def build_col_freq(self, fnames=None, row_vocab=None, col_vocab=None, multicore=True, prog_bar=True):
        """The function will treat all different word types as possible target or context words.

        Parameters
        ----------
        fnames : str, optional
            Filename of a file which records all (a user wants to process) file names of a corpus.
            Format: corpus_name + settings["fnames-ext"]
        row_vocab : :class:`~qlvl.Vocab`
            Target words (types) vocabulary.
            If a non-empty vocabulary is passed, only target words (types) in this vocab
            should be processed.
            Otherwise all possible words (types) should be processed.
        col_vocab : :class:`~qlvl.Vocab`
            Context features vocabulary.
            If a non-empty vocabulary is passed, only context features in this vocab
            should be processed.
            Otherwise all possible contexts should be processed.
        multicore : bool
            Use multicore version of the method or not.
        freqCutoff : TODO
        prog_bar : bool, optional
            Show progress bar or not.

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`
        """
        fnames = self.prepare_fnames(fnames)
        logger.info("Building collocate frequency matrix...")
        row_vocab = row_vocab if row_vocab else Vocab()
        col_vocab = col_vocab if col_vocab else Vocab()
        if multicore:
            # save word vocab and context vocab to tmp directory for subprocess usages
            # this tmp directory is 'tmp' folder inside the output path
            tmp_word_fname = os.path.join(self.tmpdir, 'word.vocab.super')
            tmp_context_fname = os.path.join(self.tmpdir, 'context.vocab.super')
            try:
                row_vocab.save(tmp_word_fname, verbose=False)
                col_vocab.save(tmp_context_fname, verbose=False)
            except Exception as err:
                logger.exception(err)

            args = (fnames, update_col_freq_caller)
            kwargs = {
                'tmpdir': self.tmpdir, 'settings': self.settings,
                'prog_bar': prog_bar,
            }
            res = None
            try:
                res = self.do_job_multicore(*args, **kwargs)
            except Exception as err:
                logger.exception(err)
            return res
        else:
            return self.do_job_single(fnames, row_vocab=row_vocab, col_vocab=col_vocab, prog_bar=prog_bar)

    def do_job_single(self, fnames, row_vocab=None, col_vocab=None, prog_bar=True):
        """Method doing job for handler class.

        Parameters
        ----------
        fnames : iterable
            A list of filenames
        row_vocab : :class:`~qlvl.Vocab`
            Target words (types) vocabulary.
        col_vocab : :class:`~qlvl.Vocab`
            Context features vocabulary.
        prog_bar

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`
        """
        pfnames = progbar(fnames, unit='file', desc='  corpus') if prog_bar else fnames
        row_vocab = row_vocab if row_vocab else Vocab()
        col_vocab = col_vocab if col_vocab else Vocab()
        matrix = defaultdict(lambda: defaultdict(int))

        for fname in pfnames:
            args = (fname,)
            kwargs = {
                'matrix': (matrix, row_vocab, col_vocab),
                'settings': self.settings,
            }
            try:
                update_col_freq(*args, **kwargs)
            except Exception as e:
                logger.exception("{} error:\n{}".format(fname, e))

        # if row_vocab.FILTERPRESENT is False, this row_vacab is constructed during processing
        # if a non-empty row_vocab is passed, its FILTERPRESENT should be True
        # which means that it should be used as a filter vocabulary for target words
        if not row_vocab.FILTERPRESENT:
            row_vocab.FILTERPRESENT = True
        # same as row_vocab
        if not col_vocab.FILTERPRESENT:
            col_vocab.FILTERPRESENT = True
        row_items = row_vocab.get_item_list()  # vocab -> a list of items (alphabetically sorted)
        col_items = col_vocab.get_item_list()
        spmatrix = transform_dict_to_spmatrix(matrix, row_items, col_items)
        return TypeTokenMatrix(spmatrix, row_items, col_items)

    def merge_results(self):
        """Merge subprocess matrices into one final matrix.
        sub-process matrices filename format: .../wcmx.sub.pid
        """
        submatrices = []
        fnames = os.listdir(self.tmpdir)
        # load all sub-process matrices
        # TODO: not load all sub-process matrices at once, save memory
        for fname in fnames:
            if 'super' in fname:  # skip super process files
                continue
            fname = os.path.join(self.tmpdir, fname)
            subMTX = TypeTokenMatrix.load(fname)
            submatrices.append(subMTX)
            os.remove(fname)  # clean tmp files
        spmx, row_items, col_items = merge_matrices(submatrices)  # spmx here is a scipy.csr_matrix
        return TypeTokenMatrix(spmx, row_items, col_items)


def update_col_freq_caller(fnames, tmpdir=None, settings=None, prog_bar=True):
    """This method will read word vocab and/or context vocab stored in tmpdir by super-process.
    Filename format of super-process objects:
        word vocab: word.vocab.super
        context vocab: context.vocab.super
    This method will save sparse matrix of sub-process.
    Filename format of sub-process objects:
        matrix: wcmx.sub.pid

    Parameters
    ----------
    fnames : iterable
        A list of filenames
    tmpdir : str
        Temporary folder
    settings : dict
    prog_bar :
    """
    if not tmpdir:
        tmpdir = os.path.join(homedir, 'tmp')
    if not settings:
        raise ValueError("Please pass a valid settings!")
    output_encoding = settings.get('output-encoding', 'utf-8')

    pid = os.getpid()
    logger.info("Starting subprocess {}".format(pid))
    pfnames = progbar(fnames, unit='file', desc='  proc({})'.format(pid)) if prog_bar else fnames

    # load word vocab and context vocab for each subprocess
    tmp_word_fname = os.path.join(tmpdir, 'word.vocab.super')
    tmp_context_fname = os.path.join(tmpdir, 'context.vocab.super')
    try:
        word_vocab = Vocab.load(tmp_word_fname, encoding=output_encoding)
        context_vocab = Vocab.load(tmp_context_fname, encoding=output_encoding)
    except Exception as err:
        logger.exception("load vocab for sub-processes error: {}".format(err))

    matrix = defaultdict(lambda: defaultdict(int))
    for fname in pfnames:
        args = (fname,)
        kwargs = {
            'matrix': (matrix, word_vocab, context_vocab),
            'settings': settings,
        }
        try:
            update_col_freq(*args, **kwargs)
        except Exception as e:
            logger.exception("{} error:\n{}".format(fname, e))

    # we don't have the word vocab and context vocab passed from the super process
    # but we can load them from the tmp directory by the super process
    word_list = word_vocab.get_item_list()
    context_list = context_vocab.get_item_list()
    spmatrix = transform_dict_to_spmatrix(matrix, word_list, context_list)
    submx = TypeTokenMatrix(spmatrix, word_list, context_list)
    tmp_proc_mtx_fname = os.path.join(tmpdir, 'wcmx.sub.{}'.format(pid))
    submx.save(tmp_proc_mtx_fname, verbose=False)


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
                # add special case when a separator line is a special case of a word line
                # e.g. `.` as a sentence separator
                isseparator = True if formatter.separator_line_machine(line) else False
                if isseparator:
                    check_right_window(win, matrix, formatter)
                    win = Window(lspan, rspan)
                cur = (match, lid)  # record current match into window
                win.update(cur)  # append it to right window
                update_cooccurrence(win, matrix, formatter)
    check_right_window(win, matrix, formatter)


class TokenManager(BaseManager):
    """Handler Class for retrieving tokens"""
    def __init__(self, queries, settings=None):
        """

        Parameters
        ----------
        queries : iterable or :class:`~qlvl.Vocab`
            Target types (queries) vocabulary. Must provide this.
            Only retrieve tokens of these types.
        settings : dict
        """
        super(TokenManager, self).__init__(settings)
        self.queries = queries
        self.tmpdir = os.path.join(self.output_path, 'tmp', 'tok.app')
        make_dir(self.tmpdir)   # create tmpdir if not exists
        clean_dir(self.tmpdir)  # clean tmpdir if exists

    @timeit
    def retrieve_tokens(self, fnames=None, col_vocab=None, multicore=True, prog_bar=True):
        """Scan/Retrieve tokens from corpus files.

        Parameters
        ----------
        fnames : str, optional
            Filename of a file which records all (a user wants to process) file names of a corpus.
            Format: corpus_name + settings["fnames-ext"]
        col_vocab : :class:`~qlvl.Vocab`
            Context features vocabulary.
            If a non-empty vocabulary is passed, only context features in this vocab
            should be processed.
            Otherwise all possible contexts should be processed.
        multicore : bool
            Use multicore version of the method or not.
        prog_bar : bool, optional
            Show progress bar or not.

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`
        """
        fnames = self.prepare_fnames(fnames)
        logger.info("Scanning tokens of queries in corpus...")
        if multicore:
            # save row_vocab and col_vocab to tmp directory for subprocess usages
            # this tmp directory is '.tmp' folder inside the output path
            tmp_type_fname = os.path.join(self.tmpdir, 'type.vocab.super')
            tmp_context_fname = os.path.join(self.tmpdir, 'context.vocab.super')
            self.queries.save(tmp_type_fname, verbose=False)
            col_vocab.save(tmp_context_fname, verbose=False)

            args = (fnames, update_tokens_caller)
            kwargs = {
                'tmpdir': self.tmpdir, 'settings': self.settings,
                'prog_bar': prog_bar,
            }
            return self.do_job_multicore(*args, **kwargs)
        else:
            return self.do_job_single(fnames, col_vocab=col_vocab, prog_bar=prog_bar)

    def do_job_single(self, fnames, col_vocab=None, prog_bar=True):
        """Method doing job for handler class.

        Parameters
        ----------
        fnames : iterable
            A list of filenames
        col_vocab : :class:`~qlvl.Vocab`
            Context features vocabulary.
        prog_bar

        Returns
        -------
        :class:`~qlvl.TypeTokenMatrix`
        """
        pfnames = progbar(fnames, unit='file', desc='  corpus') if prog_bar else fnames
        assert self.queries is not None
        row_vocab = self.queries
        col_vocab = col_vocab if col_vocab else Vocab()
        matrix = defaultdict(lambda: defaultdict(int))

        for fname in pfnames:
            args = (fname,)
            kwargs = {
                'matrix': (matrix, row_vocab, col_vocab),
                'settings': self.settings,
            }
            try:
                update_tokens(*args, **kwargs)
            except Exception as e:
                logger.exception("{} error:\n{}".format(fname, e))

        # different with ColFreqHandler.do_job_single()
        # must provide a row_vocab which is the queries vocab

        # if col_vocab.FILTERPRESENT is False, this col_vacab is constructed during processing
        # if a non-empty col_vocab is passed, its FILTERPRESENT should be True
        # which means that it should be used as a filter vocabulary for context features
        if not col_vocab.FILTERPRESENT:
            col_vocab.FILTERPRESENT = True
        row_items = sorted(matrix.keys())  # -> tokens (alphabetically ascending sorted)
        col_items = col_vocab.get_item_list()
        spmatrix = transform_dict_to_spmatrix(matrix, row_items, col_items)
        return TypeTokenMatrix(spmatrix, row_items, col_items)

    def merge_results(self):
        fnames = os.listdir(self.tmpdir)
        usenode = False
        for fname in fnames:
            # check whether using retrieve_token_nodes() or retrieve_tokens() method
            if 'super' in fname:
                if 'type2tn' in fname:
                    usenode = True
            else:
                continue
        if usenode:
            return self.merge_token_nodes()
        else:
            return self.merge_tokens()

    def merge_tokens(self):
        """Merge subprocess matrices into one final matrix.
        sub-process matrices filename format: .../tcmx.sub.pid
        """
        matrix = None
        fnames = os.listdir(self.tmpdir)
        for fname in fnames:
            if 'super' in fname:  # skip super process files
                continue
            fname = os.path.join(self.tmpdir, fname)
            subMTX = TypeTokenMatrix.load(fname)
            # TODO: improve merging matrices
            if matrix is None:
                matrix = subMTX
            else:
                matrix = matrix.merge(subMTX)
            os.remove(fname)
        return matrix

    @timeit
    def retrieve_token_nodes(self, fnames=None, multicore=True, prog_bar=True):
        fnames = self.prepare_fnames(fnames)
        logger.info("Scanning tokens of queries in corpus...")

        if multicore:
            formatter = CorpusFormatter(self.settings)
            type2tn = dict()  # type -> TypeNode
            for w in self.queries.keys():
                tn = TypeNode(type_str=w, type_fmt=formatter.type_format)
                type2tn[w] = tn
            tmp_type2tn_fname = os.path.join(self.tmpdir, 'type2tn.super')
            pickle(type2tn, tmp_type2tn_fname)

            args = (fnames, fetch_tokens_caller)
            kwargs = {
                'tmpdir': self.tmpdir, 'settings': self.settings,
                'prog_bar': prog_bar,
            }
            return self.do_job_multicore(*args, **kwargs)
        else:
            return self.retrieve_token_nodes_single(fnames)

    def retrieve_token_nodes_single(self, fnames, prog_bar=True):
        pfnames = progbar(fnames, unit='file', desc='  corpus') if prog_bar else fnames

        formatter = CorpusFormatter(self.settings)
        type2tn = dict()  # type -> TypeNode
        for w in self.queries:
            tn = TypeNode(type_str=w, type_fmt=formatter.type_format)
            type2tn[w] = tn

        for fname in pfnames:
            args = (fname,)
            kwargs = {
                'type2tn': type2tn,
                'settings': self.settings,
            }
            try:
                update_token_nodes(*args, **kwargs)
            except Exception as e:
                logger.exception("{} error:\n{}".format(fname, e))

        self.type2tn = type2tn
        return type2tn

    def merge_token_nodes(self):
        """sub-process matrices filename format: .../tcmx.sub.pid"""
        type2tn = defaultdict(list)
        fnames = os.listdir(self.tmpdir)
        for fname in fnames:
            if 'super' in fname:
                continue
            fname = os.path.join(self.tmpdir, fname)
            sub_type2tn = unpickle(fname)
            assert isinstance(sub_type2tn, dict)
            for k, v in sub_type2tn.items():
                type2tn[k].append(v)
            os.remove(fname)
        type2tn = {k: TypeNode.merge(v) for k, v in type2tn.items()}
        return type2tn


def update_tokens_caller(fnames, tmpdir=None, settings=None, prog_bar=True):
    """This method will read queries (types) vocab and/or context vocab stored in tmpdir by super-process.
    Filename format of super-process objects:
        type vocab: type.vocab.super
        context vocab: context.vocab.super
    This method will save matrix of sub-process.
    Filename format of sub-process objects:
        matrix: wcmx.sub.pid

    Parameters
    ----------
    fnames : iterable
        A list of filenames
    tmpdir : str
        Temporary folder
    settings : dict
    prog_bar :
    """
    if not tmpdir:
        tmpdir = os.path.join(homedir, 'tmp')
    if not settings:
        raise ValueError("Please pass a valid settings!")
    output_encoding = settings.get('output-encoding', 'utf-8')

    pid = os.getpid()
    logger.info("Starting subprocess {}".format(pid))
    pfnames = progbar(fnames, unit='file', desc='  proc({})'.format(pid)) if prog_bar else fnames

    # load query vocab and context vocab for each subprocess
    tmp_type_fname = os.path.join(tmpdir, 'type.vocab.super')
    tmp_context_fname = os.path.join(tmpdir, 'context.vocab.super')
    type_vocab = Vocab.load(tmp_type_fname, encoding=output_encoding)
    context_vocab = Vocab.load(tmp_context_fname, encoding=output_encoding)

    matrix = defaultdict(lambda: defaultdict(int))
    for fname in pfnames:
        args = (fname,)
        kwargs = {
            'matrix': (matrix, type_vocab, context_vocab),
            'settings': settings,
        }
        try:
            update_tokens(*args, **kwargs)
        except Exception as err:
            logger.exception("{} error:\n{}".format(fname, err))

    # save matrix of sub-process
    token_list = sorted(matrix.keys())
    context_list = context_vocab.get_item_list()
    spmatrix = transform_dict_to_spmatrix(matrix, token_list, context_list)
    submx = TypeTokenMatrix(spmatrix, token_list, context_list)
    tmp_proc_mtx_fname = os.path.join(tmpdir, 'tcmx.sub.{}'.format(pid))
    submx.save(tmp_proc_mtx_fname, verbose=False)


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
<<<<<<< HEAD:qlvl/models/handler.py
            else:  # if it's a normal line
                # add special case when a separator line is a special case of a word line
                # e.g. `.` as a sentence separator
                isseparator = True if formatter.separator_line_machine(line) else False
                if isseparator:
                    check_right_window_tok(win, fname, matrix, formatter)
                    win = Window(lspan, rspan)
=======
                # else, skip this line
            else:  # normal line
>>>>>>> develop:qlvl/bak/manager.py
                cur = (match, lid)
                win.update(cur)
                # after we update (append right) a new line to window
                # the center becomes the first item of the right buffer
                # check if the node is in words
                hit_token_for_type(win, fname, type2tn, formatter)
    process_right_window_tok(win, fname, type2tn, formatter)


class TokenScanner(BaseManager):
    """Handler Class for retrieving tokens"""
    def __init__(self, settings=None, queries=None):
        super(TokenScanner, self).__init__(settings)
        self.queries = self.types = self.words = queries
        def_key = settings['wqueries-default-key']
        # a list of word/type pairs: (specified word, neutral word)
        #self.type_pairs = [(get_word_str(w, specific=True, corpus_name=corpus_name, def_key=def_key),
        #                    get_word_str(w, specific=False, def_key=def_key))
        #                   for w in queries]
        self.vocab = None
        self.raw_vocab = None
        self.tmpdir = os.path.join(self.output_path, '.tmp', 'retr_tok')
        make_dir(self.tmpdir)
        clean_dir(self.tmpdir)

    def set_freq_list(self, vocab):
        self.raw_vocab = vocab

    def set_raw_vocab(self, vocab):
        self.raw_vocab = vocab

    def select_subvocab(self):
        freq_dict = deepcopy(self.vocab.freq_dict)
        vocab = dict()
        for sw, _ in self.type_pairs:
            wn = WordNode(self.settings, word=sw, freq=freq_dict.get(sw, 0))
            vocab[sw] = wn

    @timeit
    def retrieve_tokens(self, fnames=None, multicore=True, prog_bar=True):
        fnames = self.prepare_fnames(fnames)
        logger.info("Scanning tokens of queries in corpus...")

        if multicore:
            formatter = CorpusFormatter(self.settings)
            type2tn = dict()  # type -> TypeNode
            for w in self.queries:
                tn = TypeNode(type_str=w, type_fmt=formatter.type_format)
                type2tn[w] = tn
            tmp_type2tn_fname = os.path.join(self.tmpdir, 'type2tn.super')
            pickle(type2tn, tmp_type2tn_fname)

            args = (fnames, fetch_tokens_caller)
            kwargs = {
                'tmpdir': self.tmpdir, 'settings': self.settings,
                'prog_bar': prog_bar,
            }
            return self.do_job_multicore(*args, **kwargs)
        else:
            return self.do_batch_single(fnames)

    def do_batch_single(self, fnames, prog_bar=True):
        pfnames = progbar(fnames, unit='file', desc='  corpus') if prog_bar else fnames

        formatter = CorpusFormatter(self.settings)
        type2tn = dict()  # type -> TypeNode
        for w in self.queries:
            tn = TypeNode(type_str=w, type_fmt=formatter.type_format)
            type2tn[w] = tn

        for fname in pfnames:
            args = (fname,)
            kwargs = {
                'type2tn': type2tn,
                'settings': self.settings,
            }
            try:
                update_token_nodes(*args, **kwargs)
            except Exception as e:
                logger.exception("{} error:\n{}".format(fname, e))

        self.type2tn = type2tn
        return type2tn

    def merge_results(self):
        """sub-process matrices filename format: .../tcmx.sub.pid"""
        type2tn = defaultdict(list)
        fnames = os.listdir(self.tmpdir)
        for fname in fnames:
            if 'super' in fname:
                continue
            fname = os.path.join(self.tmpdir, fname)
            sub_type2tn = unpickle(fname)
            assert isinstance(sub_type2tn, dict)
            for k, v in sub_type2tn.items():
                type2tn[k].append(v)
            os.remove(fname)
        type2tn = {k: TypeNode.merge(v) for k, v in type2tn.items()}
        return type2tn


def fetch_tokens_caller(fnames, tmpdir=None, settings=None, prog_bar=True):
    if not settings:
        raise ValueError("Please pass a valid settings!")
    if not tmpdir:
        tmpdir = os.path.join(homedir, '.tmp')

    pid = os.getpid()
    logger.info("Starting subprocess {}".format(pid))
    pfnames = progbar(fnames, unit='file', desc='  proc({})'.format(pid)) if prog_bar else fnames

    tmp_type2tn_fname = os.path.join(tmpdir, 'type2tn.super')
    type2tn = unpickle(tmp_type2tn_fname)
    for fname in pfnames:
        args = (fname,)
        kwargs = {
            'type2tn': type2tn,
            'settings': settings,
        }
        try:
            update_token_nodes(*args, **kwargs)
        except Exception as e:
            logger.exception("{} error:\n{}".format(fname, e))

    tmp_proc_fname = os.path.join(tmpdir, 'type2tn.sub.{}'.format(pid))
    pickle(type2tn, tmp_proc_fname)


def update_tokens(filename, matrix=None, settings=None):
    """Process lines in file (filename), and match tokens.

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

    fname = os.path.basename(filename).split('.')[0]  # for filename in token
    with codecs.open(filename, 'r', input_encoding) as fin:
        lid = 0  # line number
        for line in fin:
            lid += 1
            line = line.strip()
            match = formatter.match_line(line)
            if match is None:  # if it's a separator line
                isseparator = True if formatter.separator_line_machine(line) else False
                if isseparator:
                    # process matches in the right window
                    check_right_window_tok(win, fname, matrix, formatter)
                    win = Window(lspan, rspan)
            else:  # if it's a normal line
                cur = (match, lid)
                win.update(cur)
                update_one_token(win, fname, matrix, formatter)
    check_right_window_tok(win, fname, matrix, formatter)


def check_right_window_tok(win, fname, matrix, formatter):
    """Process matches in right window.

    Parameters
    ----------
    win : :class:`~qlvl.Window`
    fname : str or bytes
    matrix : tuple
        (matrix, row_vocab, col_vocab)
    formatter : :class:`~qlvl.CorpusFormatter`
    """
    # always update window with a None first
    win.update(None)
    # case 0 : [match, ..., match] node [match, ..., match] <-- None
    # case 1 : [None, .., match, ..] node [match, ..., None] <-- None
    # case 2 : [None, ..., None] None [None, .., match, match, .., match]
    rspan = win.right_span
    for i in range(rspan):
        if win.node:
            update_one_token(win, fname, matrix, formatter)
        win.update(None)

    '''
    # handle special cases:
    # [None, ..., None] None [None, .., match, match, .., match]
    if win.node is None:
        while win.node is None:
            win.update(None)
        while win.node:
            update_one_token(win, fname, matrix, formatter)
            win.update(None)
        return

    # normal cases:
    # [None, .., match, ..] node [match, ..., None] <- None
    while win.node:
        win.update(None)
        update_one_token(win, fname, matrix, formatter)
    '''


def update_one_token(win, fname, matrix, formatter):
    """Update token and its contexts within current window.

    Parameters
    ----------
    win : :class:`~qlvl.Window`
        This is a Window object which records current items in span.
        The center item in window is the target word. And it has context
        words of left span and right span stored in two queues.
    fname : str
        Only the filename without abs path or extension.
    matrix : 3-tuple
        Includes dict of dict, row item list and column item list.
    formatter : :class:`~qlvl.CorpusFormatter`
        The getter object for fetching types, tokens and collocs.
    """
    matrix, row_vocab, col_vocab = matrix
    cnode = win.node
    if cnode is None:
        return

    match, lid = cnode[0], cnode[1]
    type_ = formatter.get_type(match)

    # row_vocab here must be used as a filter for the types
    # if type in row_vocab means that we encounter a token of query types
    found = type_ in row_vocab
    if not found:
        return

    # TODO: better to use getter.get_token() ???
    # case: node -> 'type/pos', token -> 'type/fname/lid' ???
    # token = '/'.join([type_, fname, str(lid)])
    token = formatter.get_token(match, fname, str(lid))

    # iterate 'left-span' context features
    lspan, rspan = win.left_span, win.right_span
    for i in range(lspan):
        if win.left[i] is None:
            continue
        match = win.left[i][0]  # (match, lid)
        colloc = formatter.get_colloc(match)

        if not col_vocab.FILTERPRESENT:
            col_vocab.increment(colloc)
            matrix[token][colloc] = -lspan + i  # record context's position
        elif colloc in col_vocab:
            matrix[token][colloc] = -lspan + i  # record context's position
            # matrix[token][colloc] = 1  # or just record boolean value
        else:
            pass

    for i in range(rspan):
        if win.right[i] is None:
            continue
        match = win.right[i][0]
        colloc = formatter.get_colloc(match)

        if not col_vocab.FILTERPRESENT:
            col_vocab.increment(colloc)
            matrix[token][colloc] = 1 + i  # record context's position
        elif colloc in col_vocab:
            matrix[token][colloc] = 1 + i
        else:
            pass
