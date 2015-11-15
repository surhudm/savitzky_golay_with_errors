import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.signal

import savitzky_golay_werrors

###################################################################################################

def main():

	testSine()
	#testSineEqualWeights()
	#testSpeed()

	return

###################################################################################################

def testSine():

	scatter_av = 0.1
	scatter_sigma = 0.05
	window = 15
	order = 4

	xx = np.arange(0.0, 10.0, 0.01)
	y_true = np.sin(xx)

	x = np.arange(0.0, 10.0, 0.2)
	y = np.sin(x)
	np.random.seed(250)
	q_err = np.abs(np.random.normal(scatter_av, scatter_sigma, (len(x))))
	for i in range(len(x)):
		y[i] += np.random.normal(0.0, q_err[i])

	sg = scipy.signal.savgol_filter(y, window, order, deriv = 0)
	sg_err = savitzky_golay_werrors.savgol_filter_werror(y, window, order, q_err, deriv=None)

	fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        import palettable
        ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

	ax.errorbar(x, y, yerr = q_err, fmt = '.', marker = 'o', ms = 3.0,
                label = 'Noisy data')
	ax.plot(x, sg_err, '-', label = 'This work')
	ax.plot(x, sg, '-', label = 'Traditional Savitzky-Golay filter')
	ax.plot(xx, y_true, '-', label = 'Noiseless')
	ax.legend(loc = 3, frameon=0, ncol=2)
        ax.set_ylabel("y(x)")
        ax.set_xticklabels([])

        ax = fig.add_subplot(2, 1, 2)
        ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
        ax.errorbar(x, y-np.sin(x), q_err, fmt = '.', marker = 'o', ms = 3.0)
	ax.plot(x, sg_err-np.sin(x), '-', label = 'This work')
        ax.plot(x, sg-np.sin(x), '-')
        ax.axhline(0.0, color='grey')
        ax.set_xlabel("x")
        ax.set_ylabel("y-sin(x)")

        plt.tight_layout()
	plt.savefig('Test_Sine.pdf')

	return

###################################################################################################

def testSineEqualWeights():

	scatter = 0.1
	window = 15
	order = 4

	x = np.arange(0.0, 10.0, 0.2)
	xx = np.arange(0.0, 10.0, 0.01)
	y_true = np.sin(xx)
	y = np.sin(x)
	np.random.seed(152)
	s = np.random.normal(0.0, scatter, (len(x)))
	y = y + s
	q_err = np.ones((len(x)), np.float) * scatter

	sg = scipy.signal.savgol_filter(y, window, order, deriv = 0)
	sg_err = savitzky_golay_werrors.savgol_filter_werror(y, window, order, q_err, deriv=None)

	plt.figure()
	plt.errorbar(x, y, yerr = q_err, fmt = '.', marker = 'o', ms = 3.0, label = 'data')
	plt.plot(xx, y_true, ':', label = 'True')
	plt.plot(x, sg, '-', label = 'SG')
	plt.plot(x, sg_err, '--', label = 'SG new')
	plt.legend(loc = 3)
	plt.savefig('Test_Sine_EqualWeights.pdf')

	return

###################################################################################################

def testSpeed():

	scatter = 0.1
	window = 15
	order = 4
	N = 1000

	x = np.arange(0.0, 10.0, 0.2)
	y = np.sin(x)
	np.random.seed(152)
	s = np.random.normal(0.0, scatter, (len(x)))
	y = y + s
	q_err = np.ones((len(x)), np.float) * scatter

	t1 = time.clock()
	for dummy in range(N):
		_ = scipy.signal.savgol_filter(y, window, order, deriv = 0)
	print('Numpy: %.2f s' % (time.clock() - t1))

	t1 = time.clock()
	for dummy in range(N):
		_ = savitzky_golay_werrors.savgol_filter_werror(y, window, order, q_err, deriv=None)
	print('New:   %.2f s' % (time.clock() - t1))

	return

###################################################################################################
# Trigger
###################################################################################################

if __name__ == "__main__":
	main()
