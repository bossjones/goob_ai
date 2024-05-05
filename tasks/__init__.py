# pylint: disable=wrong-import-position, wrong-import-order, invalid-name
"""
Invoke build script.
Show all tasks with::
    invoke -l
.. seealso::
    * http://pyinvoke.org
    * https://github.com/pyinvoke/invoke
"""
from __future__ import annotations

import logging
from invoke import Collection, Context, Config
from invoke import task
from .constants import ROOT_DIR, PROJECT_BIN_DIR, DATA_DIR, SCRIPT_DIR

from . import local


from . import ci
from . import view


LOGGER = logging.getLogger()

ns = Collection()

ns.add_collection(local)
ns.add_collection(ci)
ns.add_collection(view)

# https://github.com/imbrra/logowanie/blob/38a1a38ea9f5b2494e5bc986df651ff9d713fda5/tasks/__init__.py

# TODO: THINK ABOUT USING THESE MODULES https://medium.com/hultner/how-to-write-bash-scripts-in-python-10c34a5c2df1
# TODO: THINK ABOUT USING THESE MODULES https://medium.com/hultner/how-to-write-bash-scripts-in-python-10c34a5c2df1
# TODO: THINK ABOUT USING THESE MODULES https://medium.com/hultner/how-to-write-bash-scripts-in-python-10c34a5c2df1
