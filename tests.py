import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.signal
from scipy.linalg import inv
import savitzky_golay_werrors

###################################################################################################

def main():

	testSine()
	#testSineEqualWeights()
        #testSine_wsimplecov()
        testSine_wcov()
	#testSpeed()

	return

###################################################################################################

def testSine_wcov():

	scatter_av = 0.1
	scatter_sigma = 0.05
	window = 15
	order = 4

	xx = np.arange(0.0, 10.0, 0.01)
	y_true = np.sin(xx)

	x = np.arange(0.0, 10.0, 0.2)
	y = np.sin(x)
	np.random.seed(250)

	q_err = (np.abs(np.random.normal(scatter_av, scatter_sigma,
            (len(x)))))

        cov = np.diag(np.ones(q_err.size))

        # A block diagonal covariance matrix where the correlation coefficient
        # falls as a function of offset from the diagonal
        for i in range(q_err.size):
            for offset in range(1, q_err.size):
                if i+offset >= q_err.size:
                    continue
                cov[i, i+offset] = 0.3/offset**2
                cov[i+offset, i] = 0.3/offset**2

        for i in range(q_err.size):
            for j in range(i, q_err.size):
                cov[i, j] = cov[i, j] * q_err[i] * q_err[j]
                cov[j, i] = cov[i, j]

        y = np.random.multivariate_normal(y, cov)

	sg = scipy.signal.savgol_filter(y, window, order, deriv = 0)
	sg_err = savitzky_golay_werrors.savgol_filter_werror(y, window, order, cov=cov, deriv=None)

	fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.set_ylim([-2, 2])
        import palettable
        ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

	ax.errorbar(x, y, yerr = q_err, fmt = '.', marker = 'o', ms = 3.0,
                label = 'Noisy data with covariant errors')
	ax.plot(x, sg_err, '-', label = 'This work')
	ax.plot(x, sg, '-', label = 'Traditional SG')
	ax.plot(xx, y_true, '-', label = 'Noiseless')
	ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)
        ax.set_ylabel("y(x)")
        ax.set_xticklabels([])

        # Let us compute chisquare from the truth
        # Chi-squared values compared to the underlying truth
        print ("Chi-squared values compared to truth")
        chisq_trad_truth = (np.dot(np.dot((sg-np.sin(x)).T, inv(cov)), (sg-np.sin(x))))
        print ("Traditional: %.2f " % chisq_trad_truth )
        chisq_new_truth = (np.dot(np.dot((sg_err-np.sin(x)).T, inv(cov)), (sg_err-np.sin(x))))
        print ("This work: %.2f " % chisq_new_truth )
        print ("Chi-squared values compared to data")
        chisq_trad_data = (np.dot(np.dot((sg-y).T, inv(cov)), (sg-y)) )
        print ("Traditional: %.2f " % chisq_trad_data)
        chisq_new_data = (np.dot(np.dot((sg_err-y).T, inv(cov)), (sg_err-y)))
        print ("This work: %.2f " % chisq_new_data)


        ax = fig.add_subplot(2, 1, 2)
        ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
        ax.errorbar(x, (y-np.sin(x))/q_err, q_err/q_err, fmt = '.', marker = 'o', ms = 3.0)
	ax.plot(x, (sg_err-np.sin(x))/q_err, '-', label = r'This work $\chi^2=%.2f$' % chisq_new_truth)
        ax.plot(x, (sg-np.sin(x))/q_err, '-', label= "Traditional SG $\chi^2=%.2f$" % chisq_trad_truth)
        ax.axhline(0.0, color='grey')
        ax.set_xlabel("x")
        ax.set_ylabel(r"[y-sin(x)]/$\sigma_y$")
	ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)
        ax.set_ylim([-3, 3])

        plt.tight_layout()
	plt.savefig('Test_Sine_wcov.pdf')

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
        ax.set_ylim([-2, 2])
        import palettable
        ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

	ax.errorbar(x, y, yerr = q_err, fmt = '.', marker = 'o', ms = 3.0,
                label = 'Noisy data, independent errors')
	ax.plot(x, sg_err, '-', label = 'This work')
	ax.plot(x, sg, '-', label = 'Traditional SG')
	ax.plot(xx, y_true, '-', label = 'Noiseless')
	ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)
        ax.set_ylabel("y(x)")
        ax.set_xticklabels([])

        cov = np.diag(q_err**2.0)
        print ("Chi-squared values compared to data")
        chisq_trad_data = (np.dot(np.dot((sg-y).T, inv(cov)), (sg-y)) )
        print ("Traditional: %.2f " % chisq_trad_data)
        chisq_new_data = (np.dot(np.dot((sg_err-y).T, inv(cov)), (sg_err-y)))
        print ("This work: %.2f " % chisq_new_data)
        print ("Chi-squared values compared to truth")
        chisq_trad_truth = (np.dot(np.dot((sg-np.sin(x)).T, inv(cov)), (sg-np.sin(x))) )
        print ("Traditional: %.2f " % chisq_trad_truth)
        chisq_new_truth = (np.dot(np.dot((sg_err-np.sin(x)).T, inv(cov)), (sg_err-np.sin(x))))
        print ("This work: %.2f " % chisq_new_truth)

        ax = fig.add_subplot(2, 1, 2)
        ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
        ax.errorbar(x, (y-np.sin(x))/q_err, q_err/q_err, fmt = '.', marker = 'o', ms = 3.0)
	ax.plot(x, (sg_err-np.sin(x))/q_err, '-', label = 'This work $\chi^2=%.2f$' % chisq_new_truth)
        ax.plot(x, (sg-np.sin(x))/q_err, '-', label = 'Traditional SG $\chi^2=%.2f$' % chisq_trad_truth)
        ax.axhline(0.0, color='grey')
        ax.set_xlabel("x")
        ax.set_ylabel("(y-sin(x))/$\sigma_y$")
        ax.set_ylim([-3, 3])
	ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)

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
	ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)
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
