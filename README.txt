===============
datasmooth 0.61
===============

This is a scikit intended to include numerical methods for smoothing
data.  Currently, only a regularization method is available here,
which includes cross-validation for determining the optimization
parameter.  Constrained smoothing is also availalbe but requires the
cvxopt package.

Contributions of other smoothing methods are welcome.
Note that smoothing splines (which is similar to smoothing by
regularization) are available in scipy.interpolate.


Credits
-------
- Jonathan J Stickel wrote the initial code to implement smoothing by
regularization
- Tony S Yu rewrote the code using object classes.


References
----------
Comput. Chem. Eng. (2010) 34, 467

    
Installation from sources
=========================

In the directory example (the same as the file you are reading now), just do:

python setup.py install



Distribution
============

Pypi
====
Source distributions are available for download at:

http://pypi.python.org/pypi/scikits.datasmooth
