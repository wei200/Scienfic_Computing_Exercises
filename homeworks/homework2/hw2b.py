"""
Demonstation module for quadratic interpolation.
for homework 2b
modified by vv
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

def quad_interp(xi,yi):
    """
    Quadratic interpolation. Compute the coefficient of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2.
    Return c, an array containing the coefficients of 
      p(x) = c[0] + c[1]*x + c[2]*x**2
      
    """
    
    # check inputs and print error message if not valid:
    
    error_message = "xi and yi should have type numpy.ndarry"
    assert(type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message
    
    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message
    
    # Set up linear system to interpolate through data points:
    # Compute c
    A = np.vstack([np.ones(3), xi, xi**2]).T
    c = solve(A,yi)
    return c
    
    
    