#!/usr/bin/env python
"""
Smoothing by regularization of data using constraints on the smooth
solution.
"""

import numpy as np
from matplotlib.pyplot import *
from scikits import datasmooth as ds

# create simulated data
npts = 100
xmin = 0
xspan = 1
x = np.linspace(xmin,xmin+xspan,npts)
# give variability to x if desired
x = x + xspan/npts*(np.random.rand(npts)-0.5)
x[0] = xmin # force x[1] to be the defined xmin

yt = 4*x/(1+4*x)
ytp = 4/(4*x+1)**2
yt2p = -32/(4*x+1)**3
stdev = 5e-2*np.max(yt)
y = yt + stdev*np.random.randn(npts)

# perform the smoothing
d = 4
lmbd = 1e-2

# constraints:
# set point 1 at 0
Aeq = np.zeros((1,npts))
Aeq[(0,0)] = 1.0
beq = np.array([0.0])
# make sure yhat is always increasing
D = ds.derivative_matrix(x,1)
bl = np.zeros((npts-1,1))
# make sure that yhat is concave
D2 = ds.derivative_matrix(x,2)
bu2 = np.zeros((npts-2,1))
Ain = np.vstack((-D,D2))
bin = np.vstack((bl,bu2))

yh = ds.smooth_data_constr(x,y,d,lmbd,(Ain,bin),(Aeq,beq))
    
print 'scaled regularization parameter =', lmbd
print 'standard deviation =', np.std(y-yh,ddof=1)

yp = np.dot(ds.derivative_matrix(x,1),y)
y2p = np.dot(ds.derivative_matrix(x,2),y)
yhp = np.dot(ds.derivative_matrix(x,1),yh)
yh2p = np.dot(ds.derivative_matrix(x,2),yh)

figure(1)
cla()
plot(x,y,'ow',x,yh,'-b',x,yt,'-r')
legend(['scattered','smoothed','true'],loc='best',numpoints=1)
xlabel('x')
ylabel('y')
show()
draw()

figure(2)
cla()
plot(x[1:],yp,'ow',x[1:],yhp,'-b',x,ytp,'-r')
ylim( np.min(yhp)-np.abs(np.min(yhp)), np.max(yhp)*2 )
legend(['scattered','smoothed','true'],loc='best',numpoints=1)
xlabel('x')
ylabel("y'")
show()
draw()

figure(3)
cla()
plot(x[1:-1],y2p,'ow',x[1:-1],yh2p,'-b',x,yt2p,'-r')
ylim( np.min(yh2p)-np.abs(np.min(yh2p)), np.max(yh2p)+np.abs(np.min(yh2p)) )
legend(['scattered','smoothed','true'],loc='best',numpoints=1)
xlabel('x')
ylabel("y''")
show()
draw()
