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

import os
import logging
from copy import deepcopy

try:
    from configparser import ConfigParser   # Python 3
except ImportError as e:
    from ConfigParser import ConfigParser   # Python 2.6+

import nephosem

__all__ = ['ConfigLoader']

logger = logging.getLogger(__name__)

# the default configuration file should be in the directory of model
curdir = os.path.abspath(os.path.dirname(__file__))  # -> model directory
default_conf = os.path.join(curdir, 'config.ini')


class ConfigLoader(object):
    """
    Examples
    --------
    >>> from nephosem.conf import ConfigLoader
    >>> conf = ConfigLoader()  # will read default settings

    >>> new_config_fname = "/path/of/new/config/file"
    >>> settings = conf.update_config(new_config_fname)  # -> new settings based on your config file and default settings
    """

    def __init__(self, filename=None):
        """Initialize the ConfigLoader object.

        Parameters
        ----------
        filename : str
            If provided, ConfigLoader will read settings in this file other than the default 'config.ini' file
            Else, ConfigLoader will read settings in the default 'config.ini' file.
        """
        if filename is None:
            filename = default_conf
        self._settings = self.load_config(filename)

    @property
    def settings(self):
        return deepcopy(self._settings)

    @classmethod
    def read_params(cls, config, opt, sect, sett):
        try:
            sett[opt] = config.get(sect, opt)  # get parameter and its value
            if sett[opt] == -1:
                logger.warning("skip: {}".format(opt))
        except ValueError as err:
            logger.exception("Exception on {}!\n{}".format(opt, err))
            sett[opt] = None

    @classmethod
    def load_config(cls, config_file):
        """Read settings in a config file."""
        sett = dict()
        config = ConfigParser()  # Python (3) package configparser.ConfigParser
        # ConfigParser parses INI files which are expected to be parsed case-insensitively
        # disable this behaviour by replacing the RawConfigParser.optionxform() function
        config.optionxform = str
        config.read(config_file)
        sections = config.sections()  # sections in config file -> [Corpus-Format], [Span], ...

        for sect in sections:
            options = config.options(sect)  # read options/params
            for option in options:
                cls.read_params(config, option, sect, sett)
                if sect == 'Span':  # transform values of parameters of [Span] section to int
                    sett[option] = int(sett[option])

        return sett

    def update_config(self, config_file):
        """Update settings based on a config file."""
        sett = self.load_config(config_file)
        settings = deepcopy(self.settings)  # copy the default settings
        for k, v in sett.items():
            settings[k] = v

        return settings
