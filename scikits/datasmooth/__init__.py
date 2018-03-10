__version__ = '0.7.1'
try:
    from regularsmooth import *
except ImportError:
    from .regularsmooth import *
