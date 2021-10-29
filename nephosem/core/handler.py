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


"""Handler Class

Example
-------
>>>

"""

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import codecs
import logging
import os
from copy import deepcopy
import multiprocessing as mp

import qlvl
from qlvl import progbar, trange
from qlvl.core.terms import CorpusFormatter, Window
from qlvl.utils import read_fnames, read_fnames_of_corpus, make_dir, clean_dir

logger = logging.getLogger(__name__)
homedir = os.path.expanduser('~')


class Paralleler(object):
    """This is a base class of all classes that work parallel.
    This class contains a framework for parallel tasks.
    Here is the description of the procedures:
        * The main entrance is the method `process()`. It creates a job queue (for tasks sent to all sub-processes)
        and a result queue (for results returned by all sub-processes). It also creates a number of worker processes,
        each of whom will execute the method `_worker_loop()`. After creating worker processes,
        it produces the job queue by the input fnames (data). The `_worker_loop()` will process
        the tasks in the job queue and return the results in the result queue. Finally it uses
        `_process_results()` method to post process (merge) the results of each worker process
        into one final Python object.
        * The default `_job_producer()` method produces the job queue by feeding with the input
        fnames. One fname will be fetched and processed by a worker process (`_worker_loop()`).
        A sub-class inheriting `Paralleler` could over-write this method and feed other data
        into the job queue. Just ensure that one piece of data could be used by the `_do_process_job()`
        method inside the `_worker_loop()`.
        * The `_worker_loop()` method will fetch one piece of data (default, one `fname`) and
        send it to the method `_do_process_job()` and send the corresponding result to the result
        queue. Any sub-class should implement this method for different tasks.
        * The `_do_process_job` gets one piece of data, send it to a function for processing,
        and returns the result back to the `_worker_loop()` method. Any sub-class should implement
        this method.

    Attributes
    ----------
    workers : integer
        Number of CPU cores to be used parallel.
    tmpdir : str
        The path of the temporary directory used by the Class

    Notes
    -----

    """

    def __init__(self, workers=0, **kwargs):
        # the number of CPU cores used by the program
        self.workers = workers if 0 < workers < mp.cpu_count() else mp.cpu_count()-1
        # add other possible arguments to the `__dict__` of the class
        self.__dict__.update(kwargs)
        # set default tmpdir
        self.tmpdir = qlvl.tmpdir
        make_dir(self.tmpdir)
        clean_dir(self.tmpdir)

    @property
    def pid(self):
        return os.getpid()

    @property
    def subtmpdir(self):
        """When the class is instanced in a subprocess, a temporary folder for this subprocess
        will be created for its temporary files.
        """
        subtmpdir = os.path.join(self.tmpdir, str(self.pid))
        make_dir(subtmpdir)  # create the folder if not exist
        return subtmpdir

    def process(self, fnames, **kwargs):
        """Process files in the fnames list.

        Parameters
        ----------
        fnames : iterable
            This fnames is a list of file names.

        Returns
        -------
        object
            For different tasks, this method returns different objects:
                * ItemFreqHandler: returns a `Vocab` object,
                * ColFreqHandler: returns a `TypeTokenMatrix` co-occurrence frequency matrix,
                * TokenHandler: returns a Python dict mapping the type strings to their lists of tokens (`TokenNode` objects).
        """
        # define IPC manager
        manager = mp.Manager()
        # define two list (queue) for tasks and results
        job_queue = manager.Queue()
        res_queue = manager.Queue()

        # create a number of (`self.workers`) worker processes
        workers = [
            mp.Process(
                target=self._worker_loop,
                args=(job_queue, res_queue)
            )
            for _ in range(self.workers)
        ]

        # start all workers
        for worker in workers:
            worker.daemon = True  # make interrupting the process with ctrl+c easier
            worker.start()

        # produce the job queue
        self._job_producer(fnames, job_queue)

        for worker in workers:
            worker.join()

        # merge results of different sub-processes
        result = self._process_results(res_queue, n=len(fnames))
        return result

    def _job_producer(self, fnames, job_queue, **kwargs):
        """Fill the jobs queue using the input fnames.

        Each job is just a filename (Python str). One could also add other necessary objects into each job by
        overriding this method in sub-classes.

        Parameters
        ----------
        fnames : iterable of str
            A list of file names.
        job_queue : Queue of (str)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
        """
        for file_idx, fname in enumerate(fnames):
            job_queue.put(fname)
        logger.debug("putting all jobs into queue")

        # put a number ('self.workders') of None to the job queue
        for _ in range(self.workers):
            job_queue.put(None)
        logger.debug("job loop exiting, total {} jobs".format(len(fnames)))

    def _worker_loop(self, job_queue, res_queue):
        """Process the corpus, lifting batches of filename from the job queue.

        This function will be called in parallel by multiple workers (processes) to make optimal use of multicore
        machines. Each realistic sub-class should implement this method for its own task.

        Parameters
        ----------
        job_queue : Queue of (str)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
        res_queue : Queue of (int or str)
            A queue of results or indicators. When it is an integer indicator (always -1), it will be used for progress
            report. When it is a string filename, it means that there is a piece of result produced by the worker and
            saved in this temporary file.
        """
        raise NotImplementedError

    def _do_process_job(self, fname, **kwargs):
        """Process a single corpus file by the `fname`. Return a result object."""
        raise NotImplementedError

    def _process_results(self, res_queue, **kwargs):
        """Post-process (merge) the results."""
        raise NotImplementedError


class BaseHandler(Paralleler):
    """This is a base class of all handler classes.

    Contains framework for multicore method. The purpose of this class is to provide a
    reference interface for concrete handler implementations. At the same time,
    functionality that we expect to be common for those implementations is provided
    here to avoid code duplication.

    A typical procedure of processing a corpus would be

    Attributes
    ----------
    tmpindicator : str
        A string indicator for the temporary folder name used by all methods of this Class.
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
    tmpindicator = ''

    def __init__(self, settings, workers=0, **kwargs):
        super(BaseHandler, self).__init__(workers=workers, **kwargs)

        self.settings = deepcopy(settings)
        self.formatter = CorpusFormatter(self.settings)
        if 'corpus-path' not in self.settings:
            logger.warning("There is no corpus path!")
        self.corpus_path = self.settings.get('corpus-path', None)
        if 'output-path' not in self.settings:
            logger.warning("There is no output path!\nUse the default one: `~/tmp`!")
        self.output_path = self.settings.get('output-path', qlvl.tmpdir)
        if 'tmp-path' not in self.settings:
            # if 'tmp-path' is not set in the settings
            logger.warning("Not provide the temporary path!")
            if 'output-path' not in self.settings:
                # if 'output-path' is not set in the settings, use /output-path/tmp
                _tmpdir = os.path.join(self.settings['output-path'], 'tmp')
                logger.warning("Use a sub tmp directory of the output-path: '/output-path/tmp'!")
            else:
                # neither is provided, use the default one
                _tmpdir = qlvl.tmpdir  # `~/tmp`
                logger.warning("Use the default tmp directory: '~/tmp'!")
        else:
            _tmpdir = self.settings['tmp-path']
        self.tmpdir = os.path.join(_tmpdir, self.tmpindicator)  # override the one in super-class

        self.encoding = self.settings.get('file-encoding', 'utf-8')
        # TODO: choose from input_encoding, file_encoding or encoding
        self.input_encoding = self.encoding
        self.output_encoding = self.settings.get('outfile-encoding', self.encoding)

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
            # if no fnames provided, use all files inside `corpus-path` (set in settings) directory
            fnames = read_fnames_of_corpus(self.corpus_path)
        else:
            if isinstance(fnames, list):
                pass  # do nothing, already a list of (string) file names
            elif isinstance(fnames, str):
                # read file names recorded in the `fnames` ('xx.fnames') file
                fnames = read_fnames(fnames, self.encoding)
            else:
                raise ValueError("Not support other types of 'fnames' (only str or a list of strings)!")
        if len(fnames) <= 0:
            logger.warning("No corpus file selected!")
        return fnames

    def process(self, fnames, **kwargs):
        return super(BaseHandler, self).process(fnames, **kwargs)

    def _do_process_job(self, fname, **kwargs):
        raise NotImplementedError()

    def _worker_loop(self, job_queue, res_queue):
        raise NotImplementedError()

    def _process_results(self, res_queue, **kwargs):
        raise NotImplementedError()

    def update_one_file(self, filename, data, **kwargs):
        """This is the template method updating data of one corpus file."""
        # prepare variables
        input_encoding = self.settings.get('file-encoding', 'utf-8')
        lspan, rspan = self.settings['left-span'], self.settings['right-span']
        win = Window(lspan, rspan)

        # process file
        fname = os.path.basename(filename).rsplit('.', 1)[0]  # for filename in token
        with codecs.open(filename, 'r', input_encoding) as fin:
            lid = 0  # line number (1-based)
            for line in fin:
                lid += 1
                line = line.strip()
                match = self.formatter.match_line(line)
                if match is None:
                    isseparator = True if self.formatter.separator_line_machine(line) else False
                    if isseparator:
                        # process the nodes in the right window, when reaching a separator line
                        self.process_right_window(data, win, fid=fname)
                        win = Window(lspan, rspan)
                else:
                    win.update((match, lid))  # append the current match to the right window
                    # if it's a normal line, draws the type from the match
                    self.update_one_match(data, win, lid=lid, fid=fname)
        # deal with the final right window
        self.process_right_window(data, win, fid=fname)

    def process_right_window(self, *args, **kwargs):
        raise NotImplementedError

    def update_one_match(self, *args, **kwargs):
        raise NotImplementedError

