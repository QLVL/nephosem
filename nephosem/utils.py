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

from __future__ import absolute_import

"""Utility functions"""

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import os
import sys
import time
import codecs
import operator
import shutil
import json
import logging
from functools import wraps

import numpy as np
import scipy.sparse as sp

from six import iterkeys, iteritems, string_types
from six.moves import xrange

from smart_open import smart_open

if sys.version_info[0] >= 3:
    unicode = str

logger = logging.getLogger(__name__)


def timeit(fn):
    @wraps(fn)
    def timer(*args, **kwargs):
        ts = time.time()
        result = fn(*args, **kwargs)
        te = time.time()

        '''
        info = "\n************************************" + \
               "\nfunction    = {0}".format(fn.__name__) + \
               "\n  arguments = {0} {1}".format([type(arg) for arg in args], kwargs) + \
               "\n  time      = {:.4} sec".format(te - ts) + \
               "\n************************************\n"
        '''
        info = "\n************************************" + \
               "\nfunction    = {0}".format(fn.__name__) + \
               "\n  time      = {:.4} sec".format(te - ts) + \
               "\n************************************\n"
        print(info)
        return result

    return timer


def is_string(text):
    """Check if the passed text is an instance of str / unicode (Python2) or unicode / bytes (Python3)"""
    if sys.version_info > (3, 0):
        # Python 3
        return isinstance(text, str)    # or isinstance(text, bytes)
    else:
        # Python 2
        return isinstance(text, unicode) or isinstance(text, str)


def make_dir(dirname):
    """Create directory if not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # os.makedirs(dirname, exist_ok=True)  # only for Python3


def clean_dir(dirname):
    """Clean directory.
    If there are files in this directory, remove them all.
    """
    for fn in os.listdir(dirname):
        fname = os.path.join(dirname, fn)
        try:
            if os.path.isfile(fname):
                os.remove(fname)
            else:
                shutil.rmtree(fname)
        except Exception as e:
            logger.exception(e)


def sizeof(filename, suffix='B'):
    num = os.path.getsize(filename)
    return sizeof_fmt(num, suffix=suffix)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "{:3.1f}{}{}".format(num, unit, suffix)
        num /= 1024.0
    return "{:.1f}{}{}".format(num, 'Y', suffix)


def read_fnames(filename, dirname='', encoding='utf-8'):
    """
    This function reads all filenames in this provided file
    It assumes 'filename' is a string

    Parameters
    ----------
    filename : filename that records corpus filenames
    dirname : corpus directory
    encoding :

    Returns
    -------
    a list of filenames in this file
    """
    fnames = []
    # check if it is a fnames file
    basename, ext = os.path.splitext(filename)
    if ext != ".fnames":
        filename = "{}.fnames".format(filename)
    try:
        with codecs.open(filename, 'r', encoding=encoding) as inf:
            for line in inf:
                line = line.strip()
                if len(line) <= 0 or line.startswith('#'):
                    continue
                fnames.append(line)
    except IOError as e:
        raise IOError("Cannot read filenames!\n{}".format(e))

    # if filenames in this file are base names, append the corpus directory to them
    absfnames = []
    for fname in fnames:
        absfn = '{}/{}'.format(dirname, fname)
        absfn = absfn if os.sep not in fname else fname
        absfnames.append(absfn)
    fnames = absfnames
    return fnames


def read_fnames_of_corpus(corpus_path):
    """Read all file names of all files in corpus_path folder.

    Parameters
    ----------
    corpus_path : str
        The corpus path where all corpus files are located.
    """
    # TODO: skip other files which are not corpus file in corpus_path folder.
    #       such as meta file of the corpus
    fnames = []
    for root, dirs, files in sorted(os.walk(corpus_path)):
        cur_fnames = [os.path.join(root, f) for f in files]
        fnames.extend(cur_fnames)
    return fnames


def count_values(d):
    """Count the number of values in the nested dict"""
    total_vals = 0
    for k1, v1 in d.items():
        total_vals += len(v1.keys())
    return total_vals


def sort_dict(freq_dict, sorting='freq', descending=True):
    """Sort a dict by order.
    Normally if the 'sorting' is 'freq', sort the dict first by frequency descending order,
    then by alphabetic ascending order.
    If the 'sorting' is 'alpha', sort the dict by alphabetic ascending order.

    Parameters
    ----------
    freq_dict : dict
        Python dict of item to frequency pair.
    sorting : str
        'freq' for frequency order, 'alpha' for alphabetic order.
    descending : bool
        If True, sort dict by descending order of 'sorting'.
        Else, sort dict by ascending order of 'sorting'.

    Returns
    -------
    list :
        sorted_keys, a list of sorted keys of the dict
    """
    if sorting == 'freq':
        tuples = sorted(freq_dict.items(), key=operator.itemgetter(0))
        tuples = sorted(tuples, key=operator.itemgetter(1), reverse=descending)
        sorted_keys = [k for (k, _) in tuples]
        # sorted_keys = sorted(tuples, key=tuples.get, reverse=descending)
    elif sorting == 'alpha':
        sorted_keys = sorted(freq_dict.keys(), reverse=descending)
    else:
        raise ValueError("Insupportable sorting order!")

    return sorted_keys


def save_dict_json(freq_dict, filename, encoding='utf-8'):
    """Save frequency dict to json file."""
    with codecs.open(filename, 'w', encoding) as fout:
        json.dump(freq_dict, fout, ensure_ascii=False, indent=4)
        # ensure_ascii = False : show human readable characters in file.


def save_dict_plain(freq_dict, filename, encoding='utf-8', order='freq'):
    """Save frequency dict to plain txt file.

    Parameters
    ----------
    freq_dict : dict
    filename : str
    encoding : str
        'utf-8', 'latin-1', ...
    order : str
        'freq', 'alpha'

    """
    with codecs.open(filename, 'w', encoding) as fout:
        if len(freq_dict.keys()) == 0:
            return
        if order not in ['freq', 'alpha']:
            raise ValueError("Error: unsupported order!")

        tuples = sorted(freq_dict.items(), key=operator.itemgetter(0))  # alphabetic ascending
        if order == 'freq':
            tuples = sorted(tuples, key=operator.itemgetter(1), reverse=True)  # frequency descending
        for key, value in tuples:
            fout.write('{}\t{}\n'.format(key, value))


def load_dict_json(filename, encoding='utf-8'):
    """Load dict from json file."""
    with codecs.open(filename, 'r', encoding) as fin:
        freq_dict = json.load(fin)
    return freq_dict


def load_dict_plain(filename, encoding='utf-8'):
    """Load dict from plain txt file
    One word per line ( word[TAB]freq )
    """
    res_dict = dict()
    with codecs.open(filename, 'r', encoding) as fin:
        for line in fin:
            pair = line.strip().split('\t')
            if len(pair) < 2:
                continue
            res_dict[pair[0]] = float(pair[1]) if '.' in pair[1] else int(pair[1])
    return res_dict


def save_concordance(fname, typenodes, colloc_fmt='lemma', encoding='utf-8'):
    """Write out a concordance for types/concepts

    Parameters
    ----------
    fname : str
        filename to save
    colloc_fmt : str
        Options: 'lemma', 'word', 'lemma/pos', ...
    encoding : str
        default 'utf-8'
    """
    with codecs.open(fname, 'w', encoding=encoding) as fout:
        for ty in typenodes:
            for to in typenodes[ty].tokens:
                cleft = [c.to_colloc(colloc_fmt=colloc_fmt) for c in to.lcollocs]
                cright = [c.to_colloc(colloc_fmt=colloc_fmt) for c in to.rcollocs]
                fout.write("{}\t{}\t{}\t{}\n".format(str(to), ' '.join(cleft),
                    to.to_colloc(colloc_fmt=colloc_fmt), ' '.join(cright)))
                '''
                if level == 'lemma':
                    cleft = [c.to_colloc() for c in to.lcollocs]
                    rleft = [c.to_colloc() for c in to.rcollocs]
                    fout.write("{}\t{}\t{}\t{}\n".format(str(to), ' '.join(cleft), to.to_colloc(), ' '.join(rleft)))
                elif level == 'word':
                    cleft = [c.word for c in to.lcollocs]
                    rleft = [c.word for c in to.rcollocs]
                    fout.write("{}\t{}\t{}\t{}\n".format(str(to), ' '.join(cleft), to.word, ' '.join(rleft)))
                else:
                    raise ValueError("level should be either 'lemma' or 'word'!")
                '''


def pickle(obj, fname, protocol=2):
    """Pickle object `obj` to file `fname`, using smart_open so that `fname` can be on S3, HDFS, compressed etc.

    Parameters
    ----------
    obj : object
        Any python object.
    fname : str
        Path to pickle file.
    protocol : int, optional
        Pickle protocol number. Default is 2 in order to support compatibility across python 2.x and 3.x.

    """
    with smart_open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    """Load object from `fname`, using smart_open so that `fname` can be on S3, HDFS, compressed etc.

    Parameters
    ----------
    fname : str
        Path to pickle file.

    Returns
    -------
    object
        Python object loaded from `fname`.

    """
    with smart_open(fname, 'rb') as f:
        # Because of loading from S3 load can't be used (missing readline in smart_open)
        if sys.version_info > (3, 0):
            return _pickle.load(f, encoding='latin1')
        else:
            return _pickle.loads(f.read())


def read_word_queries(fname, encoding='utf-8', wquery_default_key='_DEFAULT_'):
    """
    ----------------------------------------------------------------------
    Reads a list of 'word queries' from the file 'fname'. These
    'word queries' are used in other functions as search terms for
    retrieving tokens in the corpora.
    ----------------------------------------------------------------------
    The function read_word_queries() assumes that 'fname' is a string that
    contains a file name.

    The expected file format for that file is as follows:

    - lines that start with # are ignored
    - other lines, if they don't contain tabs, are assumed to contain a
      single word representation

          e.g.:
             appel/noun

      such single word representations are represented as a string in the
      output of this function.
    - other lines, if they do contain tabs, are assumed to have the
      following format:

           corpusname:wordstr TAB corpusname:wordstr TAB etc.

          e.g.:
              LeNC:rustoord/noun TAB TwNC:rust_oord/noun

      This example indicates that what we consider to be one and the same
      word in our study, has the format "rustoord/noun" in the corpus
      LeNC and has the format "rust_oord/noun" in the corpus TwNC.
      Such information will be represented, in the output of
      readWQueriesList(), in a dictionary:

          e.g.:
                 {"LeNC": "rustoord/noun",
                  "TwNC": "rust_oord/noun",
                  "_DEFAULT_": "rustoord/noun"}

      In this example, it is assumed that the value of
      'settings["wqueries-default-key"]' is "_DEFAULT_".
      There always is a key 'settings["wqueries-default-key"]' in these
      output dictionaries. If the input line does not explicitly contain
      a corpus name equal to 'settings["wqueries-default-key"]', then the
      first value/word in the input line is also used as the value/word for
      'settings["wqueries-default-key"]'.
    ----------------------------------------------------------------------
    [format of result]
    - The result is a list of items, with each of these items either being
      a string (which indicates that the same search term can be used in
      all corpora) or a dictionary that maps corpus names onto queries/words
      and that maps 'settings["wqueries-default-key"]' onto the default
      query/word.
    ----------------------------------------------------------------------

    Parameters
    ----------
    fname :
    encoding : str
        default 'utf-8'

    Returns
    -------
    list of dicts

    """
    retval = []
    if not os.path.isfile(fname):
        return retval
    with codecs.open(fname, 'r', encoding=encoding) as inf:
        for line in inf:
            line = line.strip()
            if len(line) <= 0 or line.startswith('#'):
                continue

            if '\t' not in line:
                retval.append(line)
            else:
                item = {
                    wquery_default_key: None,
                }
                fields = line.split("\t")
                for field in fields:
                    eles = field.split(":")
                    if len(eles) != 2:
                        continue
                    item_corpus_name = eles[0].strip()
                    item_word_str = eles[1].strip()
                    item[item_corpus_name] = item_word_str
                    if item[wquery_default_key] is None:
                        item[wquery_default_key] = item_word_str
                retval.append(item)
    return retval


def get_word_str(wquery, specific=False, corpus_name='', def_key=None):
    """

    Parameters
    ----------
    wquery
        can be a string or a dictionary
        the result of getWordStr() always is a string
    specific
        refers to whether or not a corpus specific word
        representation is requested.
    corpus_name
        only used if 'specific=True'
    def_key
        settings['wqueries-default-key']

    Returns
    -------

    """
    if not isinstance(wquery, dict):
        return wquery

    if specific and corpus_name in wquery:
        retval = wquery[corpus_name]
    else:
        retval = wquery[def_key]
    return retval


class SaveLoad(object):
    """Serialize/deserialize object from disk, by equipping objects with the save()/load() methods.

    Warnings
    --------
    This uses pickle internally (among other techniques), so objects must not contain unpicklable attributes
    such as lambda functions etc.

    """
    @classmethod
    def load(cls, fname, mmap=None):
        """Load an object previously saved using :meth:`~gensim.utils.SaveLoad.save` from a file.

        Parameters
        ----------
        fname : str
            Path to file that contains needed object.
        mmap : str, optional
            Memory-map option.  If the object was saved with large arrays stored separately, you can load these arrays
            via mmap (shared memory) using `mmap='r'.
            If the file being loaded is compressed (either '.gz' or '.bz2'), then `mmap=None` **must be** set.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.save`
            Save object to file.

        Returns
        -------
        object
            Object loaded from `fname`.

        Raises
        ------
        AttributeError
            When called on an object instance instead of class (this is a class method).

        """
        logger.info("loading %s object from %s", cls.__name__, fname)

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        logger.info("loaded %s", fname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        """Load attributes that were stored separately, and give them the same opportunity
        to recursively load using the :class:`~gensim.utils.SaveLoad` interface.

        Parameters
        ----------
        fname : str
            Input file path.
        mmap :  {None, ‘r+’, ‘r’, ‘w+’, ‘c’}
            Memory-map options. See `numpy.load(mmap_mode)
            <https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.load.html>`_.
        compress : bool
            Is the input file compressed?
        subname : str
            Attribute name. Set automatically during recursive processing.

        """
        def mmap_error(obj, filename):
            return IOError(
                'Cannot mmap compressed object %s in file %s. ' % (obj, filename) +
                'Use `load(fname, mmap=None)` or uncompress files manually.'
            )

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s", attrib, cfname, mmap)
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with np.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = np.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = np.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = np.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None", attrib)
            setattr(self, attrib, None)

    @staticmethod
    def _adapt_by_suffix(fname):
        """Get compress setting and filename for numpy file compression.

        Parameters
        ----------
        fname : str
            Input filename.

        Returns
        -------
        (bool, function)
            First argument will be True if `fname` compressed.

        """
        compress, suffix = (True, 'npz') if fname.endswith('.gz') or fname.endswith('.bz2') else (False, 'npy')
        return compress, lambda *args: '.'.join(args + (suffix,))

    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2):
        """Save the object to a file. Used internally by :meth:`gensim.utils.SaveLoad.save()`.

        Parameters
        ----------
        fname : str
            Path to file.
        separately : list, optional
            Iterable of attributes than need to store distinctly.
        sep_limit : int, optional
            Limit for separation.
        ignore : frozenset, optional
            Attributes that shouldn't be store.
        pickle_protocol : int, optional
            Protocol number for pickle.

        Notes
        -----
        If `separately` is None, automatically detect large numpy/scipy.sparse arrays in the object being stored,
        and store them into separate files. This avoids pickle memory errors and allows mmap'ing large arrays back
        on load efficiently.

        You can also set `separately` manually, in which case it must be a list of attribute names to be stored
        in separate files. The automatic check is not performed in this case.

        """
        logger.info("saving %s object under %s, separately %s", self.__class__.__name__, fname, separately)

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol,
                                       compress, subname)
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            # restore attribs handled specially
            for obj, asides in restores:
                for attrib, val in iteritems(asides):
                    setattr(obj, attrib, val)
        logger.info("saved %s", fname)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves :class:`~gensim.utils.SaveLoad` instances.

        Parameters
        ----------
        fname : str
            Output filename.
        separately : list or None
            List of attributes to store separately.
        sep_limit : int
            Don't store arrays smaller than this separately. In bytes.
        ignore : iterable of str
            Attributes that shouldn't be stored at all.
        pickle_protocol : int
            Protocol number for pickle.
        compress : bool
            If True - compress output with :func:`numpy.savez_compressed`.
        subname : function
            Produced by :meth:`~gensim.utils.SaveLoad._adapt_by_suffix`

        Returns
        -------
        list of (obj, {attrib: value, ...})
            Settings that the caller should use to restore each object's attributes that were set aside
            during the default :func:`~gensim.utils.pickle`.

        """
        asides = {}
        sparse_matrices = (sp.csr_matrix, sp.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in iteritems(self.__dict__):
                if isinstance(val, np.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)

        # whatever's in `separately` or `ignore` at this point won't get pickled
        for attrib in separately + list(ignore):
            if hasattr(self, attrib):
                asides[attrib] = getattr(self, attrib)
                delattr(self, attrib)

        recursive_saveloads = []
        restores = []
        for attrib, val in iteritems(self.__dict__):
            if hasattr(val, '_save_specials'):  # better than 'isinstance(val, SaveLoad)' if IPython reloading
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname, attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore, pickle_protocol, compress, subname))

        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in iteritems(asides):
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))

                elif isinstance(val, (sp.csr_matrix, sp.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(
                            subname(fname, attrib, 'sparse'),
                            data=val.data,
                            indptr=val.indptr,
                            indices=val.indices
                        )
                    else:
                        np.save(subname(fname, attrib, 'data'), val.data)
                        np.save(subname(fname, attrib, 'indptr'), val.indptr)
                        np.save(subname(fname, attrib, 'indices'), val.indices)

                    data, indptr, indices = val.data, val.indptr, val.indices
                    val.data, val.indptr, val.indices = None, None, None

                    try:
                        # store array-less object
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = data, indptr, indices
                else:
                    logger.info("not storing attribute %s", attrib)
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except Exception:
            # restore the attributes if exception-interrupted
            for attrib, val in iteritems(asides):
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]

    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2):
        """Save the object to a file.

        Parameters
        ----------
        fname_or_handle : str or file-like
            Path to output file or already opened file-like object. If the object is a file handle,
            no special array handling will be performed, all attributes will be saved to the same file.
        separately : list of str or None, optional
            If None, automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This prevent memory errors for large objects, and also allows
            `memory-mapping <https://en.wikipedia.org/wiki/Mmap>`_ the large arrays for efficient
            loading and sharing the large arrays in RAM between multiple processes.

            If list of str: store these attributes into separate files. The automated size check
            is not performed in this case.
        sep_limit : int, optional
            Don't store arrays smaller than this separately. In bytes.
        ignore : frozenset of str, optional
            Attributes that shouldn't be stored at all.
        pickle_protocol : int, optional
            Protocol number for pickle.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.load`
            Load object from file.

        """
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object", self.__class__.__name__)
        except TypeError:  # `fname_or_handle` does not have write attribute
            self._smart_save(fname_or_handle, separately, sep_limit, ignore, pickle_protocol=pickle_protocol)
