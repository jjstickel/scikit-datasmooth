__version__ = 0.5

try:
    from regularsmooth import *
except ImportError as error:
    print error
    print "Constrained smoothing is disabled."
    from regularsmooth_no_constr import *

