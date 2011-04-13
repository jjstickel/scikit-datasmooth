#!/usr/bin/env python
"""
Smoothing by regularization of ordered data using midpoint-rule
integration and matching standard deviation
"""

import numpy as np
from matplotlib.pyplot import *
from scikits import datasmooth as ds

# create simulated data
npts = 100
xmin = 0
xspan = 2*np.pi
x = np.linspace(xmin,xmin+xspan,npts)
# give variability to x if desired
x = x + xspan/npts*(np.random.rand(npts)-0.5)

yt = np.sin(x)
stdev = 1e-1*np.max(yt)
y = yt + stdev*np.random.randn(npts)

# perform the smoothing
d = 4
yh,lmbd = ds.smooth_data(x,y,d,stdev=stdev,midpointrule=True)

print 'scaled regularization parameter =', lmbd
print 'standard deviation =', np.std(y-yh,ddof=1)

cla()
plot(x,y,'ow',x,yh,'-b',x,yt,'-r')
legend(['data','smoothed','true'], numpoints=1, loc='best')
show()
draw()
