# Savitzky Golay filter for data with errorbars

A Savitzky–Golay filter (Savitzky, A.; Golay, M.J.E., 1964, "Smoothing and
Differentiation of Data by Simplified Least Squares Procedures") is often
applied to data for the purpose of smoothing the data without greatly distorting
the signal.

This kind of filter has been often used in the scientific literature. However
almost all data inherently comes with noise, and the noise properties can differ
from point to point. This python script improves upon the traditional
Savitzky-Golay filter by accounting for errors or covariance in the data.

The inputs and arguments are all modelled after scipy.signal.savgol_filter
