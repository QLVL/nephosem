import codecs
import logging
import multiprocessing as mp
import os
import threading
from copy import deepcopy
from collections import defaultdict

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from qlvl import progbar, trange
from qlvl.core.vocab import Vocab
from qlvl.core.matrix import TypeTokenMatrix
from qlvl.core.handler import BaseHandler
from qlvl.core.graph import SentenceGraph, MacroGraph
from qlvl.specutils import mxutils

logger = logging.getLogger(__name__)
homedir = os.path.expanduser('~')


class DepRelHandler(BaseHandler):
    """Handler Class for processing dependency relations"""

    def __init__(self, settings, workers=0, targets=None, features=None, **kwargs):
        super(DepRelHandler, self).__init__(settings, workers=workers)
        # you could make the program check only the targets and/or features you provide
        if targets is not None:
            if isinstance(targets, list):
                self.targets = Vocab( {t: 0 for t in targets} )
            elif isinstance(targets, Vocab):
                self.targets = targets
            else:
                raise AttributeError("The `targets` should be a list of string or a `Vocab` object!")
        else:
            self.targets = Vocab()
        # normally, we don't set features previously
        self.macros = []

    def read_templates(self, fname=None, macros=None, encoding='utf-8'):
        """Read the templates from a CSV/TSV file.
        The file has lines of content like the following (including a header):
            ID	Target Regex	Feature Regex	Tareget Description	Feature Description	ID
            1	(?P<LEMMA>\w+)/(?P<POS>N)\w*	<-(?P<DEPREL>nsubj)$ (?P<LEMMA>\w+)/(?P<POS>V)\w*	noun	subject of verb	1

        Parameters
        ----------
        fname : str, optional
            File name of the templates file.
        macros : iterable of :class:`~qlvl.core.graph.TemplateGraph`, optional
            TemplateGraph instances when not passing the file name.
        encoding : str, default 'utf-8'
            File encoding of the template file.

        Raises
        ------
        ValueError
            If either of the `fname` or `templates` is not provided.

        Examples
        --------
        >>> dephan = DepRelHandler(settings)
        >>> template_fname = "{}/tests/data/DependencyFeatureTemplates.subgroup.tsv".format(qlvl.rootdir)
        >>> dephan.read_templates(fname=template_fname)
        >>> dephan.templates[0]
        <-(?P<DEPREL>nsubj)$ (?P<LEMMA>\w+)/(?P<POS>V)\w*
        >>> templates = deepcopy(dephan.templates)
        >>> dephan.read_templates(macros=templates)
        >>> dephan.templates[0]
        <-(?P<DEPREL>nsubj)$ (?P<LEMMA>\w+)/(?P<POS>V)\w*

        """
        if macros:
            self.macros = macros
        elif fname:
            self.macros = MacroGraph.read_csv(fname)  # TODO: use different encodings
        else:
            raise ValueError("Please provide either fname or features!")

    def build_dependency(self, fnames=None):
        """Build a dependency frequency matrix for corpus files provided.

        Parameters
        ----------
        fnames : str, optional
            Path of file recording corpus file names ('fnames' file of a corpus).
            If this is provided, only the files recorded in this fnames file would be processed.
            Else, all files and folders inside the 'corpus-path' of settings would be processed.
        targets : list of str or :class:`~qlvl.core.vocab.Vocab`, optional
            Target types/words to process.
            If this is provided, only process these targets when matching the sentence with macros.
            Else, all possible targets would be checked when matching sentences.

        Returns
        -------
        features : iterable of :class:`~qlvl.core.graph.TemplateGraph`

        Examples
        --------
        # create a DepRelHandler instance
        >>> freqMTX = dephan.build_dependency()
        >>> freqMTX
        [11, 24]  ->[agent_/->[nsubjpass_boy/N],_apple/N]  ->[agent_/->[nsubjpass_girl/N],_apple/N]  ->[nsubj_apple/N]  ->[nsubj_boy/N]  ->[nsubj_girl/N]  <-nsubj_/V->[acomp_healthy/JJ]  <-nsubj_/V->[acomp_old/JJ]  ...
        apple/N   NaN                                      NaN                                       NaN                NaN              NaN               3                               NaN                         ...
        ask/V     NaN                                      NaN                                       NaN                1                NaN               NaN                             NaN                         ...
        be/V      NaN                                      NaN                                       4                  2                NaN               NaN                             NaN                         ...
        boy/N     NaN                                      NaN                                       NaN                NaN              NaN               1                               1                           ...
        eat/V     1                                        1                                         NaN                7                6                 NaN                             NaN                         ...
        girl/N    NaN                                      NaN                                       NaN                NaN              NaN               1                               NaN                         ...
        give/V    NaN                                      NaN                                       NaN                3                1                 NaN                             NaN                         ...
        ...       ...                                      ...                                       ...                ...              ...               ...                             ...                         ...

        """
        fnames = self.prepare_fnames(fnames)
        logger.info("Building dependency features...")
        _res = self.process(fnames)

        logger.info("Building matrix...")
        self.freqMTX = self.build_matrix_by_matches()
        return self.freqMTX

    def build_matrix_by_matches(self):
        """Build a frequency matrix by the matches."""
        freq_dict = defaultdict(lambda: defaultdict(int))
        targets, contexts = set(), set()
        for macro in self.macros:
            for i in range(len(macro.matched_nodes)):
                trgt = macro.target(index=i)
                targets.add(trgt)
                feat = macro.feature(index=i)
                contexts.add(feat)
                freq_dict[trgt][feat] += 1
        targets = sorted(targets)
        contexts = sorted(contexts)

        freqmx = mxutils.transform_dict_to_spmatrix(freq_dict, targets, contexts)
        freqMTX = TypeTokenMatrix(freqmx, targets, contexts)
        return freqMTX

    def process(self, fnames, **kwargs):
        """
        queue_factor : int, optional
            Multiplier for size of queue -> size = number of workers * queue_factor.
        """
        return super(DepRelHandler, self).process(fnames, **kwargs)

    def _worker_loop(self, job_queue, res_queue):
        """Worker loop function which gets one by one job from the job queue.
        This function would be executed by many threads.

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
            # Vocab object cannot be pickled, so we have to delete it from macro attributes
            for macro in tally:
                del macro.target_filter
            res_queue.put(tally)

        logger.debug("worker exiting")

    def _do_process_job(self, fname, **kwargs):
        """Match every sentence in the file by calling `update_dep_rel()` function
        and return the list of matched features.

        Parameters
        ----------
        fname : str
            One corpus file name

        Returns
        -------
        features : iterable
            The matched features from sentences in the `fname` file.
        """
        macros = deepcopy(self.macros)
        # when you provide targets to filter, add it to every macro
        # speed up by matching target-feature pattern
        # replace the target regular expression of patterns with the targets
        if self.targets is not None:
            for macro in macros:
                macro.target_filter = self.targets.deepcopy()
        self.update_dep_rel(fname, macros)
        return macros

    def update_dep_rel(self, fname, macros, **kwargs):
        """This is the real method that is used for processing!!!
        Procedures:
        1. read sentences from the corpus file
        2. for each sentence, match every template/pattern
            2.1 if targets are provided, only match the sentence which satisfies the targets
            2.2 so the matching should be a target-feature matching
            2.3 this is a way of speeding up the process when the targets are provided
        """
        # read each sentence from the corpus file
        end_bound = self.settings['separator-line-machine']
        sentences = read_sentence(fname, formatter=self.formatter, encoding=self.input_encoding)
        for s in sentences:
            ss = SentenceGraph(sentence=s, formatter=self.formatter)
            for macro in macros:
                ss.match_pattern(macro)
        return

    def _process_results(self, res_queue, n=0):
        """Get all results (matched features) from result queue and merge them."""
        for _ in trange(n):
            res = res_queue.get()
            for i in range(len(res)):
                feat = res[i]
                self.macros[i].matched_nodes.extend(feat.matched_nodes)
                self.macros[i].matched_edges.extend(feat.matched_edges)


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
    generator of sentences (tuple(int, string))
    """
    with codecs.open(filename, 'r', encoding=encoding) as fin:
        sentence = []
        lid = 0
        for line in fin:
            lid += 1
            line = line.strip()
            match = formatter.match_line(line)  # a valid line
            if match:
                sentence.append((lid, line))  # add line index
            # if line.startswith(end_bound):  # end of a sentence
            if formatter.separator_line_machine(line):
                yield sentence
                sentence = []
        if len(sentence) > 0:  # if file does not end with a '</s' line
            yield sentence
