import os
import logging
from copy import deepcopy

try:
    from configparser import ConfigParser   # Python 3
except ImportError as e:
    from ConfigParser import ConfigParser   # Python 2.6+

import qlvl

__all__ = ['ConfigLoader']

logger = logging.getLogger(__name__)

# the default configuration file should be in the directory of model
curdir = os.path.abspath(os.path.dirname(__file__))  # -> model directory
default_conf = os.path.join(curdir, 'config.ini')


class ConfigLoader(object):
    """
    Examples
    ========
    >>> from qlvl.conf import ConfigLoader
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

    def updateConfig(self, config_file):
        # duplicate and deprecated method
        logger.info("Please use `update_config()`, this one is deprecated!")
        return self.update_config(config_file)

    def update_config(self, config_file):
        """Update settings based on a config file."""
        sett = self.load_config(config_file)
        settings = deepcopy(self.settings)  # copy the default settings
        for k, v in sett.items():
            settings[k] = v

        return settings
