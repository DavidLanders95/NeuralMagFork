# SPDX-License-Identifier: MIT

VERSION = "0.9.3"

import neuralmag.common.logging as logging
from neuralmag.common import *
from neuralmag.field_terms import *
from neuralmag.loggers import *
from neuralmag.solvers import *

logging.info_green(f"[NeuralMag] Version {VERSION}")

__all__ = common.__all__ + field_terms.__all__ + loggers.__all__ + solvers.__all__
