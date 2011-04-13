"""
Smooth data by regularization.

Implementation Notes
--------------------
    Smooth data by regularization as described in [1]. Optimal values
    for the regularization parameter, lambda, can be calulated using
    the generalized cross-validation method described in [2] or by
    constraining the standard deviation between the smoothed and
    measured data as described in [3]. Both methods for calculating
    lambda are reviewed in [1].

    Smoothing with constraints is also implemented, but without the
    features for determining an optimal value for the regularizaiton
    parameter. Requires the cvxopt module (constrained smoothing is
    disabled if cvxopt is not installed).

References
----------
    [1] Comput. Chem. Eng. (2010) 34, 467
    [2] Anal. Chem. (2003) 75, 3631
    [3] AIChE J. (2006) 52, 325
"""

#Copyright (c) 2010, Jonathan Stickel
#
#All rights reserved.
#
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#
#  Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the
#  distribution.
#
#  Neither the name of Jonathan Stickel nor the names of any
#  contributors may be used to endorse or promote products derived
#  from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from scipy.linalg import solve
from scipy import optimize

# if cvxopt is not installed, skip code related to constrained smoothing
try:
    import cvxopt
    from cvxopt import solvers as cvxslvrs
    __all__ = ['smooth_data', 'smooth_data_constr', 'fmin_options','calc_derivative', 'derivative_matrix']
    cvxslvrs.options['show_progress'] = False    
    incl_const = True
except ImportError:
    print('The module "cvxopt" is not installed.  Constrained smoothing will not be available.')
    __all__ = ['smooth_data', 'fmin_options','calc_derivative', 'derivative_matrix']    
    incl_const = False


fmin_options = dict(disp=False, maxfun=50, xtol=1e-2, ftol=1e-6)

def is_even(val):
    return np.mod(val, 2) == 0

def as_array1d(a):
    return np.asarray(a).ravel()

def dot_array1d(mat, a):
    return np.dot(np.asarray(mat), np.asarray(a)).ravel()

def derivative_matrix(x, d=1):
    """Return order `d` derivative matrix."""
    num_pts = len(x)
    if d == 0:
        return np.identity(num_pts)
    dx = x[d:] - x[:-d]
    V = np.diag(1.0 / dx)
    D_diff = np.diff(derivative_matrix(x, d - 1), axis=0)
    D = d * np.dot(V, D_diff)
    #return np.asmatrix(D)
    return D

def calc_derivative(x, y, d=1):
    """Return order `d` derivative"""
    D = derivative_matrix(x, d=d)
    return dot_array1d(D, y)


def smooth_data(x, y, d=2, lmbd=None, derivative=0, xhat=None, stdev=None, 
                lmbd_guess=1.0, weights=None, relative=False, 
                midpointrule=False):
    """Return smoothed y-values and optimal regularization parameter.
    
    Smooths the `y`-values of 1D data by Tikhonov regularization. Also, return 
    scaled regularization parameter `lmbd` used for smoothing when not provided.
    
    If neither the regularization parameter, `lmbd`, nor the standard deviation 
    `stdev` are provided, then use generalized cross-validation to determine the 
    optimal value for the regularization parameter.
    
    Parameters
    ----------
    x : 1D array
        Data x-values.
    y : 1D array
        Data y-values.

    Optional Parameters
    -------------------
    xhat : 1D array
        A vector of x-values to use for the smooth curve; must be monotonically
        increasing.
    derivative : int
        Return derivative of given order. The given value is added to the 
        smoothing derivative, `d`, such that the returned derivative has the 
        smoothness given by the input to `d`.
    d : int
        Derivative used to calculate roughness. When `d = 2`, the 2nd derivative 
        (i.e. the curvature) of the data is used to calculate roughness.
    stdev : float
        When provided, determine optimal value for lambda by matching the 
        provided value with the standard deviation of yhat-y; if the option 
        'relative' is True, then a relative standard deviation is inferred.
    lmbd : float
        Scaled regularization parameter; larger values give smoother results.
    lmbd_guess : float
        The initial value for lambda used in the iterative minimization
        algorithm to find the optimal value.
    weights : 1D array
        A vector of weighting values for fitting each point in the data.
    relative : bool
        Use relative differences for the goodness of fit term.
    midpointrule : bool
        Use the midpoint rule for the integration terms rather than a
        direct sum; this option conflicts with the option "xhat".

    Returns
    -------
    y_smooth : 1D array
        The smooth y-values
    lmbd : float, optional
        'Optimal' scaled regularization parameter. Returned unless it is 
        provided as an input parameter.

    Example
    --------
    >>> import numpy as np
    >>> import scikits.datasmooth as ds
    >>> import matplotlib.pyplot as plt
    >>> npts = 100
    >>> x = np.linspace(0,2*np.pi,npts)
    >>> y = np.sin(x)
    >>> y = y + 1e-1*np.random.randn(npts)
    >>> yh,lmbd = ds.smooth_data (x, y, d=4, stdev=1e-1)
    >>> plt.plot(x,y,'o',x,yh)
    
    """
    if xhat is not None and midpointrule:
        print('warning: midpointrule is currently not used if xhat is provided '
              '(since x, y may be scattered)')
        midpointrule = False
    return_lambda = lmbd is None
    d += derivative
    data = Data(x, y, xhat)
    matrices = RegularizationMatrices(data, d, weights, relative, midpointrule)
    if lmbd is None:
        lmbd = optimal_lambda(lmbd_guess, data, matrices, stdev, relative)
    y_smooth = regularize_data(lmbd, data, matrices)
    return_val = y_smooth
    if not derivative == 0:
        return_val = calc_derivative(data.xhat, y_smooth, derivative)
    if return_lambda:
        return return_val, lmbd
    return return_val


def regularize_data(lmbd, data, matrices):
    A, b = matrices.build_linear_system(lmbd, data)
    yhat = solve(A, b)
    return as_array1d(yhat)

def optimal_lambda(lmbd_guess, data, matrices, stdev, relative):
    if stdev is None:
        args = (data, matrices)
        return fmin(variance_gcv, lmbd_guess, args)
    else:
        args = (data, matrices, stdev, relative)
        return fmin(variance_std, lmbd_guess, args)

def fmin(func, x0, args=None):
    """Minimize given function starting with given initial guess
    
    This function wraps `scipy.optimize.fmin`; adjust keyword arguments using
    module level variable, `fmin_options`.
    """
    log_x0 = np.log10(x0)
    output = optimize.fmin(func, log_x0, args, full_output=True, **fmin_options)
    log_x = output[0][0]
    warnflag = output[4]
    if warnflag:
        print ('Warning: maximum number of function evaluations exceeded.')
    return 10**log_x

def variance_gcv(log10lmbd, data, matrices):
    """Return squared variance from generalized cross-validation.
    
    Solve for optimal lambda by using this function with fmin.
    """
    # TODO: implement Eiler's partial H computation for large datasets
    lmbd = 10**np.asscalar(log10lmbd)
    A, b = matrices.build_linear_system(lmbd, data)
    yhat = solve(A, b)
    M = matrices.M
    y = matrices.as_column_vector(data.y)
    H = M * solve(A, M.T * matrices.W)
    y_diff = M * yhat - y
    variance = y_diff.T * y_diff / data.N / (1 - np.trace(H)/data.N)**2
    return variance

def variance_std(log10lmbd, data, matrices, stdev, relative):
    """Return squared difference between the standard deviation of (y - yhat)
    
    Solve for optimal lambda by using this function with fmin.
    """
    lmbd = 10**np.asscalar(log10lmbd)
    yhat = regularize_data(lmbd, data, matrices)
    y_diff = dot_array1d(matrices.M, yhat) - data.y
    if relative:
        y_diff = y_diff / data.y.astype(float)
    stdevd = np.std(y_diff)
    return (stdevd - stdev)**2


class Data(object):
    """Data structure for storing `x`, `y` data."""
    
    def __init__(self, x, y, xhat=None):
        self.x = x
        self.y = y
        if xhat is None:
            xhat = x
        self.xhat = xhat
        self.N = x.size
        self.Nhat = xhat.size
        self._validate_data()
    
    def _validate_data(self):
        if not self.x.size == self.y.size:
            raise ValueError('x and y must be equal length 1D arrays')
        if not all(np.diff(self.xhat)>0):
            if np.alltrue(self.xhat == self.x):
                raise ValueError('x must be monotonically increasing if a '
                                 'separate xhat is not provided')
            else:
                raise ValueError('xhat must be monotonically increasing')
        # this check is not needed anymore with new linear mapping code, JJS 4/8/11
        #if self.x.min() < self.xhat.min() or self.xhat.max() < self.x.max():
        #    raise ValueError('xhat must at least span the data')


class RegularizationMatrices(object):
    
    def __init__(self, data, d, weights, relative, midpointrule):
        self.d = d
        self._weights = weights
        self._relative = relative
        self._midpointrule = midpointrule
        self.D = self.derivative_matrix(data, d)
        self.M = self._mapping_matrix(data)
        self.U, self.W = self._weight_matrices(data)
    
    def as_column_vector(self, x):
        return np.asmatrix(x).T
    
    def build_linear_system(self, lmbd, data):
        M = self.M
        D = self.D
        U = self.U
        W = self.W
        y = self.as_column_vector(data.y)
        delta = float(np.trace(D.T*D)) / float(data.Nhat**(2+self.d))
        A = M.T * W * M + lmbd * delta**(-1) * D.T * U * D
        b = M.T * W * y
        return A, b
        
    def derivative_matrix(self, data, d):
        """Return order `d` derivative matrix."""
        D = derivative_matrix(data.xhat, d)
        return np.asmatrix(D)

    def _mapping_matrix(self, data):
        """Linear interpolation mapping matrix, which maps `yhat` to `y`."""
        # map the scattered points to the appriate index of the smoothed points
        idx = np.searchsorted(data.xhat,data.x,'right') - 1
        # allow for "extrapolation"; i.e. for xhat extremum to be interior to x
        idx[idx==-1] += 1
        idx[idx==data.Nhat-1] += -1
        # create the linear interpolation matrix
        M2 = (data.x - data.xhat[idx])/(data.xhat[idx+1] - data.xhat[idx])
        M1 = 1 - M2
        j = range(data.N)
        M = np.zeros((data.N,data.Nhat))
        M[j,idx[j]] = M1
        M[j,idx[j]+1] = M2
        return np.asmatrix(M)

    def _weight_matrices(self, data):
        if self._weights is not None:
            W = np.diag(self._weights)
        else:
            W = np.identity(data.N)
        if self._relative:
            Yinv = np.diag(1.0 / data.y)
            W = np.dot(W, Yinv**2)
        if self._midpointrule:
            U, W = self._apply_midpoint_rule(data, W)
        else:
            U = np.identity(data.Nhat - self.d)
        return np.asmatrix(U), np.asmatrix(W)
    
    def _apply_midpoint_rule(self, data, W):
        B = self._integration_matrix(data)
        W = np.dot(W, B)
        U = self._submatrix_of_integration_matrix(B)
        return U, W

    def _integration_matrix(self, data):
        """Return integration matrix based on the midpoint rule."""
        ones = np.ones(data.N - 1)
        Bhat = np.diag(-ones, -1) + np.diag(ones, 1)
        Bhat[0, 0] = -1
        Bhat[-1, -1] = 1
        B = np.diag(np.dot(Bhat, data.x)) / 2.0
        return B

    def _submatrix_of_integration_matrix(self, B):
        """Return integration matrix whose size matches derivative matrix."""
        d = self.d
        start = np.floor(d/2)
        end = -start if is_even(d) else -(start + 1)
        b_slice = slice(start, end)
        return B[b_slice, b_slice]


if incl_const: # constrained smoothing code that depends on cvxopt
    def smooth_data_constr(x, y, d, lmbd, inequality=None, equality=None, xhat=None, 
                         weights=None, relative=False, midpointrule=False):
        """
        Smooths y vs. x values by Tikhonov regularization. This version
        assumes that constraints are provided, hence a quadratic program
        iterative method is used to find the solution.  In addition to x
        and y, required input paramters includes the smoothing derivative
        d and the regularization parameter lmbd.  Determination of the
        optimal regularization parameter is not implemented.

        Optional parameters
        -------------------
        inequality : tuple of numpy arrays (Ain, bin)
            Inequality constraints, i.e. Ain*yhat <= bin.
        equality : tuple of numpy arrays (Aeq, beq)
            Equality constraints, i.e. Aeq*yhat = beq.
        xhat : 1D array
            A vector of x-values to use for the smooth curve; must be
            monotonically increasing.
        weights : 1D array
            A vector of weighting values for fitting each point in the data.
        relative : bool
            Use relative differences for the goodnes of fit term
        midpointrule : bool
            Use the midpoint rule for the integration terms rather than a
            direct sum; this option conflicts with the option "xhat"

        Returns
        -------
        yhat : 1D array
            The smooth y-values
        """
        Ain, bin, Aeq, beq = (None,)*4
        if inequality is not None:
            Ain = float_matrix(inequality[0])
            bin = float_matrix(inequality[1])
        if equality is not None:
            Aeq = float_matrix(equality[0])
            beq = float_matrix(equality[1])

        if xhat is not None and midpointrule:
            print ('warning: midpointrule is currently not used if xhat is '
                   'provided (since x,y may be scattered)')
            midpointrule = False
        if xhat is None:
            xhat = x
        data = Data(x, y, xhat)
        matrices = CRegularizationMatrices(data, d, weights, relative, midpointrule)
        A, b = matrices.build_linear_system(lmbd, data)

        sol = cvxslvrs.qp(A, -b, Ain, bin, Aeq, beq)
        if not sol['status'] == 'optimal':
            print ('Warning: the solution did not fully converge')

        return np.asarray(sol['x'])[:,0]


    def float_matrix(val): 
        return cvxopt.matrix(val.astype(float))

    class CRegularizationMatrices(RegularizationMatrices):
        def as_column_vector(self, x):
            return cvxopt.matrix(x.copy())
        def derivative_matrix(self, data, d):
            D = super(CRegularizationMatrices, self).derivative_matrix(data, d)
            return cvxopt.matrix(D)
        def _mapping_matrix(self, data):
            M = super(CRegularizationMatrices, self)._mapping_matrix(data)
            return cvxopt.matrix(M)
        def _weight_matrices(self, data):
            U, W = super(CRegularizationMatrices, self)._weight_matrices(data)
            return cvxopt.matrix(U), cvxopt.matrix(W)

#if __name__ == '__main__':
#    import nose
#    nose.runmodule('test_regression.py')
