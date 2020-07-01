FULL_MATRIX_MODE = True

from . import graph_model
from . import optimizer
from . import utils
from . import np_framework

if FULL_MATRIX_MODE:
    from . import operators_full_matrix
else:
    from . import operators