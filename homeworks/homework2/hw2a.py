"""
Demostration script for quadratic interpolation.
for homework2 of UWHPSC course
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

# Set up linear syster to interpolate through data points:

# Data points:
xi = np.array([-1.,1.,2])
yi = np.array([0.,4.,3])

# Define A and b for Ax = b
A = np.array([[1.,-1.,1.],[1.,0.,0.],[1.,2.,4.]])
b = yi

# Solve Ax = b
c = solve(A,b)

print "The polynomial coefficients are:"
print c

# Plot the resulting polynomial
x = np.linspace(-2,3,1001)
y = c[0]+c[1]*x+c[2]*x**2

# Plot the figure
plt.figure(1)

# Clear the figure
plt.clf()

# connect points with a blue line
plt.plot(x,y,'b-')

# Add data points (polynomial should go through these points!)
plt.plot(xi,yi,'ro')
plt.ylim(-2,8) #set limits in y 

plt.title("Data points and interpolating polynomial")
plt.savefig('demo1plot.png')    #save figure as .png file