from __future__ import absolute_import, print_function, unicode_literals

# from six import text_type as str
# from six import binary_type as bytes
# from builtins import str
import os

# check whether ipython(jupyter) notebook is using this module
in_ipynb = True
try:
    get_ipython
except NameError:
    in_ipynb = False

if in_ipynb:
    from tqdm import tqdm_notebook as progbar
    from tqdm import tnrange as trange
else:
    from tqdm import tqdm as progbar
    from tqdm import trange as trange
# tqdm.monitor_interval = 0

from .conf import *
from .logging import *
from .core import *
from .models import *
from .specutils import *

rootdir = os.path.dirname(__file__)
tmpdir = os.path.join(os.path.expanduser('~'), 'tmp')
