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

import os
import sys
import logging
import tempfile

try:
    import threading
except ImportError as e:
    import dummy_threading as threading


_log_state = threading.local()
_log_state.indentation = 0

_python_version = sys.version_info[0]


__all__ = ['DefaultFormatter']


class DefaultFormatter(logging.Formatter):
    """This class inherits `logging.Formatter` and is used for setting the format of a handler."""

    err_fmt = "ERROR: %(msg)s"
    wrn_fmt = "WARNING: %(msg)s"
    dbg_fmt = "DEBUG: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(msg)s"

    def __init__(self):
        fmt_ = "%(levelno)d: %(msg)s"
        if _python_version < 3:
            super(DefaultFormatter, self).__init__(fmt_)
        else:
            super(DefaultFormatter, self).__init__(fmt=fmt_, datefmt=None, style='%')

    def format(self, record):

        if _python_version < 3:
            # Save the original format configured by the user
            # when the logger formatter was instantiated
            format_orig = self._fmt

            # Replace the original format with one customized by logging level
            if record.levelno == logging.DEBUG:
                self._fmt = DefaultFormatter.dbg_fmt

            elif record.levelno == logging.INFO:
                self._fmt = DefaultFormatter.info_fmt

            elif record.levelno == logging.ERROR:
                self._fmt = DefaultFormatter.err_fmt

            # Call the original formatter class to do the grunt work
            result = logging.Formatter.format(self, record)

            # Restore the original format configured by the user
            self._fmt = format_orig
        else:
            format_orig = self._style._fmt
            if record.levelno == logging.DEBUG:
                self._style._fmt = DefaultFormatter.dbg_fmt
            elif record.levelno == logging.WARNING:
                self._style._fmt = DefaultFormatter.wrn_fmt
            elif record.levelno == logging.INFO:
                self._style._fmt = DefaultFormatter.info_fmt
            elif record.levelno == logging.ERROR:
                self._style._fmt = DefaultFormatter.err_fmt

            result = logging.Formatter.format(self, record)
            self._style._fmt = format_orig

        return result


def init_logging():
    """This function creates the root logger of the qlvl module.
    This logger has one stream handler (shown in console) and one file handler (in file `~/tmp/qlvl.log`).
    All other loggers created in sub-modules / files will inherit these two handlers.
    The logger has the message level `logging.INFO`. While the stream handler has the message level `logging.DEBUG`,
    and the file handler has the message level `logging.WARNING`.
    Since the logger has the `INFO` level, those `DEBUG` messages would not be sent to the handlers.
    So although the stream handler has the `DEBUG` level, it would not show any `DEBUG` message.
    To let the stream handler show `DEBUG` messages, just use `logger.setLevel(logging.DEBUG)` to lower the level of
    the logger. Then the `DEBUG` messages would be sent from the logger to the handlers.
    However, since the file handler has the `WARNING` level, this action does not affect it.

    Notes
    -----
    Python logging level: ERROR > WARNING > INFO > DEBUG
    """
    # ------- set default logger and add a console handler and a file handler -------
    default_formatter = DefaultFormatter()

    # create a stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(default_formatter)
    console_handler.setLevel(logging.DEBUG)

    # on linux systems, temporary folder is `/tmp`, `/var/tmp`, `/usr/tmp` in order
    # tmpdir = tempfile.gettempdir()
    # do not use system temporary folder, users might not have privilege.
    homedir = os.path.expanduser('~')
    tmpdir = os.path.join(homedir, 'tmp')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # create a file handler
    file_handler = logging.FileHandler('{}/qlvl.log'.format(tmpdir))  # ~/tmp/qlvl.log
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                                       datefmt='%d/%m/%Y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.WARNING)

    # logger for the whole module
    logger = logging.getLogger('qlvl')
    if len(logger.handlers) > 0:
        # if there are handlers in this logger, clean them
        logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def create_file_handler(log_fname):
    """Create a file handler whose file name is the given string.

    Parameters
    ----------
    log_fname : str
        The file name of the logging file handler.

    Returns
    -------
    file_handler : logging handler
        Logging file handler
    """
    file_handler = logging.FileHandler(log_fname)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                                       datefmt='%d/%m/%Y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.WARNING)
    return file_handler


def change_log_file(logger, fname):
    """Change filename for saving log"""
    logger.handlers.pop()
    fhd = create_file_handler(fname)
    logger.addHandler(fhd)


init_logging()
