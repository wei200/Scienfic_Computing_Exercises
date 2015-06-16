
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
    
    
def test_quad1():
        """
        Test code, no return value or exception if test runs properly.
        """
        xi = np.array([-1.,0.,2.])
        yi = np.array([1.,-1.,7.])
        c = quad_interp(xi,yi)
        c_true = np.array([-1.,0.,2.])
        print "c =       ", c
        print "c_true =  ",c_true
        assert np.allclose(c, c_true),\
            "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)
            

def plot_quad(xi,yi):
    """
    calls quad_interp to compute c and plot both the interpolating polynomial and
    the data points,save the resulting figure as quadratic.png
    """
    c = quad_interp(xi,yi)
    x = np.linspace(xi.min()-1, xi.max()+1,1000)
    y = c[0]+c[1]*x+c[2]*x**2
    
    # Plot the figure
    plt.figure(1)

    # Clear the figure
    plt.clf()

    # connect points with a blue line
    plt.plot(x,y,'b-')

    # Add data points (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')

    plt.title("Data points and interpolating polynomial")
    plt.savefig('quadratic.png')    #save figure as .png file
    
    
def test_quad2():
    # Generate  test2 by specifying c_true first:
    c_true = np.array([7., 2., -3.])
    # Points to interpolate:
    xi = np.array([-1.,  0.,  2.])
    # Function values to interpolate:
    yi = c_true[0] + c_true[1]*xi + c_true[2]*xi**2
    
    c = quad_interp(xi,yi)
    print "c=   ", c
    print "c_true =   ",c_true
    
    assert np.allclose(c,c_true),\
        "Incorrect result, c = %s, Expected:c = %s " % (c,c_true)
    
    plot_quad(xi,yi)
    

def cubic_interp(xi,yi):
    """
    solve the cubic interpolation with len=4 input
    """
    
    # check if inputs are valid
    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message
    
    error_message = " xi and yi should have length 4"
    assert len(xi)==4 and len(yi)==4, error_message
    
    # set up the linear system equation
    A = np.vstack([np.ones(4), xi, xi**2,xi**3]).T
    c = solve(A,yi)       
    return c


def plot_cubic(xi,yi):
    """
    calls quad_interp to compute c and plot both the interpolating polynomial and
    the data points,save the resulting figure as quadratic.png
    """
    c = cubic_interp(xi,yi)
    x = np.linspace(xi.min()-1, xi.max()+1,1000)
    y = c[0]+c[1]*x+c[2]*x**2+c[3]*x**3
    
    # Plot the figure
    plt.figure(1)

    # Clear the figure
    plt.clf()

    # connect points with a blue line
    plt.plot(x,y,'b-')

    # Add data points (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')

    plt.title("Data points and interpolating polynomial")
    plt.savefig('cubic.png')    #save figure as .png file
    

def test_cubic1():
    # Generate a test by specifying c_true first:
    c_true = np.array([7., -2., -3., 1.])
    # Points to interpolate:
    xi = np.array([-1.,  0.,  1., 2.])
    # Function values to interpolate:
    yi = c_true[0] + c_true[1]*xi + c_true[2]*xi**2 + c_true[3]*xi**3
    
    c = cubic_interp(xi,yi)
    print "c=   ", c
    print "c_true =   ",c_true
    
    assert np.allclose(c,c_true),\
        "Incorrect result, c = %s, Expected:c = %s " % (c,c_true)
    
    plot_cubic(xi,yi)

def poly_interp(xi,yi):
    # check if inputs are valid
    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message
    
    error_message = " xi and yi should have length 4"
    assert len(xi) == len(yi), error_message
    
    # set up the linear system equation
    n=len(xi)
    A = np.vstack([xi**j for j in range(n)]).T
    c = solve(A,yi)       
    return c
    
def plot_poly(xi,yi):
    n = len(xi)
    c = poly_interp(xi,yi)
    x = np.linspace(xi.min()-1, xi.max()+1,1000)
    y = c[n-1]
    for j in range(n-1,0,-1):
        y = y*x + c[j-1]
    
    # Plot the figure
    plt.figure(1)

    # Clear the figure
    plt.clf()

    # connect points with a blue line
    plt.plot(x,y,'b-')

    # Add data points (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')
    plt.ylim(yi.min()-1,yi.max()+1)

    plt.title("Data points and interpolating polynomial")
    plt.savefig('poly.png')    #save figure as .png file
    
    
def test_poly1():
    # Generate a test by specifying c_true first:
    c_true = np.array([7., -2., -3., 1.])
    # Points to interpolate:
    xi = np.array([-1.,  0.,  1., 2.])
    # Use Horner's rule:
    n = len(xi)
    yi = c_true[n-1]
    for j in range(n-1, 0, -1):
        yi = yi*xi + c_true[j-1]
    # Now interpolate and check we get c_true back again.
    c = poly_interp(xi,yi)
    
    print "c=    ",c
    print "c_true=  ",c_true
    assert np.allclose(c,c_true),\
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)
    plot_poly(xi,yi)

def test_poly2():
    # Generate a test by specifying c_true first:
    c_true = np.array([0., -6., 11., -6., 1.])
    # Points to interpolate:
    xi = np.array([-1.,  0.,  1., 2., 4.])
    # Use Horner's rule:
    n = len(xi)
    yi = c_true[n-1]
    for j in range(n-1, 0, -1):
        yi = yi*xi + c_true[j-1]
    # Now interpolate and check we get c_true back again.
    c = poly_interp(xi,yi)
    
    print "c=    ",c
    print "c_true=  ",c_true
    assert np.allclose(c,c_true),\
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)
    plot_poly(xi,yi)
    
           
if __name__=="__main__":
    # "main program"
        print "Running test..."
        test_quad1()
        test_quad2()
        test_cubic1()
        test_poly1()
        test_poly2()