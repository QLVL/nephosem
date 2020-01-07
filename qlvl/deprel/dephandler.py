import codecs
import logging
import multiprocessing as mp
import os
import threading
from copy import deepcopy

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from .basic import PathTemplate, SentenceGraph

from qlvl import progbar, trange
from qlvl.core.handler import BaseHandler
from qlvl.core.terms import CorpusFormatter
from qlvl.utils import make_dir, clean_dir, pickle, unpickle

logger = logging.getLogger(__name__)
homedir = os.path.expanduser('~')


class DepRelManager(BaseHandler):
    """Handler Class for processing dependency relations"""
    def __init__(self, settings):
        super(DepRelManager, self).__init__(settings)
        self.templates = []
        self.features = []

        self.tmpdir = os.path.join(self.output_path, 'tmp', 'dep.rel')
        make_dir(self.tmpdir)  # create tmpdir if not exists
        clean_dir(self.tmpdir)  # clean tmpdir if exists

    def read_template(self, fname=None, features=None, encoding='utf-8'):
        """Read paths from file"""
        if features:
            self.features = features
            return features
        elif fname:
            templates = []
            with codecs.open(fname, encoding=encoding) as fin:
                for line in fin:
                    eles = line.strip().split(":")
                    nodes, edges = [], []
                    for i in range(0, len(eles), 2):
                        nodes.append(eles[i])
                    for i in range(1, len(eles) - 1, 2):
                        edges.append(eles[i])
                    templates.append(PathTemplate(nodes, edges))
            self.templates = templates
            return templates
        else:
            raise ValueError("Please provide either fname or templates!")

    def read_features(self, fname=None, features=None, encoding='utf-8'):
        if features:
            self.features = features
            return features
        elif fname:
            pass
        else:
            raise ValueError("Please provide either fname or features!")

    def process(self, fnames, queue_factor=2):
        """

        Parameters
        ----------
        fnames
        queue_factor : int, optional
            Multiplier for size of queue -> size = number of workers * queue_factor.
        """
        job_queue = Queue(maxsize=queue_factor * self.workers)
        # progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue,))
            for _ in range(self.workers)
        ]

        workers.append(threading.Thread(
            target=self._job_producer,
            args=(fnames, job_queue),
            kwargs={}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        return

    def _worker_loop(self, job_queue):
        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                break
            fname = job

            tally = self._do_process_job(fname, thread_private_mem)

            jobs_processed += 1

        logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def _job_producer(self, fnames, job_queue):

        for file_idx, fname in enumerate(fnames):
            job_queue.put(fname)

        for _ in range(self.workers):
            job_queue.put(None)
        logger.debug("job loop exiting, total %i jobs", len(fnames))

    def _do_process_job(self, fname, thread_private_mem):
        res = update_dep_rel(fname, self.features, settings=self.settings)

    def _get_thread_working_mem(self):
        return 1.0

    def build_dep_rel(self, fnames=None, multicore=True):
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
        """
        fnames = self.prepare_fnames(fnames)
        logger.info("Building dependency relations...")
        if multicore:
            # save word vocab and context vocab to tmp directory for subprocess usages
            # this tmp directory is 'tmp' folder inside the output path
            tmp_temp_fname = os.path.join(self.tmpdir, 'path.template.super')
            try:
                pickle(self.features, tmp_temp_fname)
            except Exception as err:
                logger.exception(err)

            args = (fnames, update_dep_rel_caller)
            kwargs = {'tmpdir': self.tmpdir, 'settings': self.settings}
            res = None
            try:
                res = self.do_job_multicore(*args, **kwargs)
            except Exception as err:
                logger.exception(err)
            return res
        else:
            return self.do_job_single(fnames)

    def do_job_single(self, fnames, **superkwargs):
        """Method doing job for handler class.

        Parameters
        ----------
        fnames : iterable
            A list of filenames
        """
        pfnames = progbar(fnames, unit='file', desc='  corpus')

        for fname in pfnames:
            args = (fname,)
            kwargs = {'features': self.features, 'settings': self.settings}
            try:
                update_dep_rel(*args, **kwargs)
            except Exception as e:
                logger.exception("{} error:\n{}".format(fname, e))

    def merge_results(self):
        """Merge subprocess matrices into one final matrix.
        sub-process matrices filename format: .../matches.sub.pid
        """
        feature_matches = []
        fnames = os.listdir(self.tmpdir)
        # load all sub-process matrices
        # TODO: not load all sub-process matrices at once, save memory
        for fname in fnames:
            if 'super' in fname:  # skip super process files
                continue
            fname = os.path.join(self.tmpdir, fname)
            subfeatures = unpickle(fname)
            feature_matches.append(subfeatures)
            os.remove(fname)  # clean tmp files

        for feats in feature_matches:
            for i, feat in enumerate(feats):
                self.features[i].matched_nodes.extend(feat.matched_nodes)
                self.features[i].matched_edges.extend(feat.matched_edges)

        '''
        paths = [[] for _ in self.templates]
        for submatches in path_matches:
            for i, pmatches in enumerate(submatches):
                paths[i].extend(pmatches.matches)
        paths = [Path(temp, matches=m) for temp, m in zip(self.templates, paths)]
        return paths
        '''


def update_dep_rel_caller(fnames, tmpdir=None, settings=None):
    """This method will save path template matches of sub-process.
    Filename format of sub-process objects:
        matrix: paths.sub.pid

    Parameters
    ----------
    fnames : iterable
        A list of filenames
    tmpdir : str
        Temporary folder
    settings : dict
    """
    if not tmpdir:
        tmpdir = os.path.join(homedir, 'tmp')
    if not settings:
        raise ValueError("Please pass a valid settings!")

    pid = os.getpid()
    logger.info("Starting subprocess {}".format(pid))
    pfnames = progbar(fnames, unit='file', desc='  proc({})'.format(pid))

    # load path templates for each subprocess
    tmp_temp_fname = os.path.join(tmpdir, 'path.template.super')
    try:
        features = unpickle(tmp_temp_fname)
    except Exception as err:
        logger.exception("load path templates for sub-processes error: {}".format(err))
        return

    for fname in pfnames:
        args = (fname,)
        kwargs = {'features': features, 'settings': settings}
        try:
            update_dep_rel(*args, **kwargs)
        except Exception as e:
            logger.exception("{} error:\n{}".format(fname, e))
    tmp_proc_matches_fname = os.path.join(tmpdir, 'matches.sub.{}'.format(pid))
    pickle(features, tmp_proc_matches_fname)

    '''
    paths = [Path(temp) for temp in features]
    for tid, m in matches:
        paths[tid].add_path(m)
    tmp_proc_matches_fname = os.path.join(tmpdir, 'matches.sub.{}'.format(pid))
    pickle(paths, tmp_proc_matches_fname)
    '''
<<<<<<< HEAD


def update_dep_rel(filename, features=None, settings=None):
    input_encoding = settings.get('file-encoding', 'utf-8')
    formatter = CorpusFormatter(settings)

    # read each sentence from the corpus file
    sentences = read_sentence(filename, formatter=formatter, encoding=input_encoding)
    res = []
    sid = 0
    for s in sentences:
        ss = SentenceGraph(sentence=s.split('\n'))
        for f in features:
            ss.match_feature(f)
        '''
        g = Graph(sentence=ss)
        for i, p in enumerate(features):
            valid_matches = g.match(p)
            for m in valid_matches:
                # [pid, sid, node1, node2, ...] -> path id, sentence id, nodes...
                tmp = (i, m)
                res.append(tmp)
        sid += 1
        # if sid % 10000 == 0:
        #     print(sid, end=' ')
        '''
    return res


def read_sentence(filename, formatter=None, encoding='utf-8'):
    """Read sentences from corpus file.

    Parameters
    ----------
    filename : str
    formatter : qlvl.CorpusFormatter
    encoding : str
        default 'utf-8'

    Returns
    -------
    generator of sentences (string)
    """
    with codecs.open(filename, 'r', encoding=encoding) as fin:
        sentence = []
        for line in fin:
            line = line.strip()
            sentence.append(line)
            match = formatter.match_line(line)
            # add Mariana
            if (match):
                eles = formatter.settings['line-format'].split(',')
                ele2id = {e: match.group(i + 1) for i, e in enumerate(eles)}
                eles = [ele2id[x] for x in ['word', 'pos', 'lemma', 'curId', 'toId', 'rel']]
                sentence.append('\t'.join(eles))
            # end Mariana
            if line.startswith('</s'):  # end of a sentence
                yield '\n'.join(sentence)
                sentence = []
        if len(sentence) > 0:  # if file does not end with a '</s' line
            yield '\n'.join(sentence)
=======
>>>>>>> develop
