#!/usr/bin/env python
"""
Smoothing by regularization of scattered data using
generalized-cross-validation
"""

import numpy as np
from matplotlib.pyplot import *
from scikits import datasmooth as ds

# create simulated data
npts = 50
xmin = 0
xspan = 2*np.pi
x = xmin + xspan*np.random.rand(npts)

yt = np.sin(x)
stdev = 1e-1*np.max(yt)
y = yt + stdev*np.random.randn(npts)

# perform the smoothing
d = 4
Nhat = 200
xmin = np.min(x)
xmax = np.max(x)
xh = np.linspace(xmin-0.1,xmax+0.1,Nhat)

yh,lmbd = ds.smooth_data(x,y,d,xhat=xh)

yht = np.sin(xh)

print 'scaled regularization parameter =', lmbd

cla()
plot(x,y,'ow',xh,yh,'-b',xh,yht,'-r')
legend(['scattered','smoothed','true'],loc='best',numpoints=1)
show()
draw()
