import numpy as np
from scipy.linalg import svd, inv

'''
Simple Savitzky Golay interpolator, assuming the data has errorbars.
Utilizes numpy.polyfit
Author: Surhud More (Kavli IPMU)
Email bug reports to surhudkicp [at] gmail.com

TODO:
    Python 3 compatibility
    Check that window_length and degree are integers
    Checks that y and error are arrays
'''
def savgol_filter_werror(y, window_length, degree, error, cov=None, deriv=None):
    ynew = y * 0.0

    if cov is not None:
        yorig = y * 1.0

        # Q weight Q^T = Cinv
        # where the columns of Q are the eigenvectors of Cinv
        # and weight is a diagonal matrix with eigenvalues of Cinv
        # Chi^2 = (y - ymodel)^T Cinv (y-ymodel)
        #       = (y - ymodel)^T Q weight Q^T (y-ymodel)
        # where ymodel = (a0 a1 a2 .. an)(  1    1    1    1  )
        #                                ( x1   x2   x3   x4  )
        #                                ( ..   ..   ..   ..  )
        #                                ( x1^n x2^n x3^n x4^n)
        # Define ynew = Q^T(y), and find polyfit solution to ynew, with weight
        # Then the solution from polyfit is:
        # (a0' a1' ... an') = Q^T (a0 a1 a2 .. an)
        #  So,
        #  (a0 a1 a2 .. an) = Q (a0' a1' ... an')
        covinv = inv(cov)
        weight, Q = eigh(covinv)

        y = np.dot(np.transpose(Q), y)
    else:
        weight = 1./error**2

    # Check that window_length is odd
    if window_length % 2 == 0:
        print("Window length must be odd\n")
        exit(11)

    # Take care that the window does not spill out of our array
    margin = int(window_length/2)
    xarr = np.arange(-margin, margin+1)
    for i in range(margin, y.size-margin):
        # Obtain solution using polyfit
        z = np.polyfit(xarr, y[i-margin:i+margin+1], deg=degree, w=weight[i-margin:i+margin+1])

        # Backout real solution if covariance is present
        if cov is not None:
            z = np.dot(Q, z)
        p = np.poly1d(z)
        if deriv is not None:
            p = np.polyder(p, m=deriv)
        ynew[i] = p[0]

    # Now fit the left boundary, by fitting the first window_length points with
    # a degree order polynomial
    z = np.polyfit(xarr, y[:window_length], deg=degree, w=weight[:window_length])
    if cov is not None:
        z = np.dot(Q, z)
    p = np.poly1d(z)
    if deriv is not None:
        p = np.polyder(p, m=deriv)
    for i in range(margin):
        ynew[i] = p(xarr[i])

    # Now fit the right boundary, by fitting the last window_length points with
    # a degree order polynomial
    z = np.polyfit(xarr, y[-window_length:], deg=degree, w=weight[-window_length:])
    if cov is not None:
        z = np.dot(Q, z)
    p = np.poly1d(z)
    if deriv is not None:
        p = np.polyder(p, m=deriv)
    for i in range(margin):
        ynew[y.size-margin+i] = p(xarr[i+margin+1])

    return ynew
