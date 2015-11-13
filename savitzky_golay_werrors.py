import numpy as np

'''
Simple Savitzky Golay interpolator, assuming the data has errorbars.
Utilizes numpy.polyfit
Author: Surhud More (Kavli IPMU)
Email bug reports to surhudkicp [at] gmail.com

TODO:
    Python 3 compatibility
    Add support for covariance
    Check that window_length and degree are integers
    Checks that y and error are arrays
'''
def savgol_filter_werror(y, window_length, degree, error, deriv=None):
    ynew = y * 0.0

    weight = 1./error**2
    # Now check that window_length is odd
    if window_length % 2 == 0:
        print("Window length must be odd\n")
        exit(11)

    # Only change xnew where the window length does not fall over
    margin = int(window_length/2)
    xarr = np.arange(-margin, margin+1)
    for i in range(margin, y.size-margin):
        # Now obtain solution using polyfit
        z = np.polyfit(xarr, y[i-margin:i+margin+1], deg=degree, w=weight[i-margin:i+margin+1])
        p = np.poly1d(z)
        if deriv is not None:
            p = np.polyder(p, m=deriv)
        ynew[i] = p[0]

    # Now fit the left boundary, by fitting the first window_length points with a
    # degree order polynomial
    z = np.polyfit(xarr, y[:window_length], deg=degree, w=weight[:window_length])
    p = np.poly1d(z)
    if deriv is not None:
        p = np.polyder(p, m=deriv)
    for i in range(margin):
        ynew[i] = p(xarr[i])

    # Now fit the right boundary, by fitting the last window_length points with a
    # degree order polynomial
    z = np.polyfit(xarr, y[-window_length:], deg=degree, w=weight[-window_length:])
    p = np.poly1d(z)
    if deriv is not None:
        p = np.polyder(p, m=deriv)
    for i in range(margin):
        ynew[y.size-margin+i] = p(xarr[i+margin+1])

    return ynew
