__version__ = 0.61
try:
    from regularsmooth import *
except ImportError:
    from .regularsmooth import *
