#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

N_LS = 100
LS_SIZE = 10**3

RANGE_BEGIN = -9
RANGE_END = 9
RANGE_SIZE = 1000

N_SAMPLES_ESTIMATE = 10**5

def target_func(xs, sigma=1):
    """
    Given a matrix of samples and their single attribute 'x', return a matrix of
    values computed as '(sin(x) + cos(x)) * x² + e' where 'e ~ N(0, 1)' is a
    noise variable.

    Parameters
    ----------
    xs : array-like, shape = [n_samples, 1]
         The samples

    Returns
    -------
    ys : array-like, shape = [n_samples, 1]
         The target values.
    """

    es = np.random.normal(0, 1, xs.shape)
    ys = (np.sin(xs) + np.cos(xs)) * xs * xs + es
    return ys

def estimate_error(f, regrs, x):
    """
    Estimates the residual error, the squared bias, the variance and the
    expected mean square error of the given regressions 'regrs', for a function
    'f()' at the given point 'x'.

    Does this by generating a large number of samples for the value 'x' using
    'f()'.
    """

    # Generates 'N_SAMPLES_ESTIMATE' samples by running the function 'f()' on
    # point 'x'.

    xs = np.empty((N_SAMPLES_ESTIMATE, 1))
    xs.fill(x)
    ys = f(xs)

    #
    # Computes the residual error by computing the standard deviation of the
    # sample set generated on point 'x'.
    #

    res_error = np.var(ys)

    #
    # Computes the squared bias by computed the squared difference of the
    # mean of all the samples, and the mean of the average output of the trained
    # regressions on the samples.
    #

    mu_ys = np.mean(ys)

    ys_regrs = np.array([regr.predict(x)[0] for regr in regrs])

    mu_ys_regrs = np.mean(ys_regrs)

    sq_bias = np.mean((mu_ys - mu_ys_regrs)**2)
    var = np.mean((ys_regrs - mu_ys_regrs)**2)

    error = res_error + sq_bias + var

    # Alternative way of computing the error rate, by testing the regressions
    # using the 'mean_squared_error()' function:
    # xs2 = np.empty((N_LS, 1))
    # xs2.fill(x)
    # error = mean_squared_error(f(xs2), ys_regrs)

    return (res_error, sq_bias, var, error)

def estimate_errors(regr_t, f, n_ls=N_LS, ls_size=LS_SIZE):
    """
    Trains 'n_ls' regressions on 'n_ls' randomly generated training set of size
    'ls_size' and estimate their errors with 'estimate_error()' on a range of
    'x' values determined by 'RANGE_BEGIN', 'RANGE_END' and 'RANGE_SIZE'.
    """

    xs = np.random.uniform(RANGE_BEGIN, RANGE_END, (ls_size, 1))

    def train_regression():
        regr = regr_t()
        regr.fit(xs, f(xs))
        return regr

    regrs = [train_regression() for i in range(0, n_ls)]

    xs = np.linspace(RANGE_BEGIN, RANGE_END, RANGE_SIZE)
    errs = np.vectorize(lambda x: estimate_error(f, regrs, x))(xs)

    return xs, np.array(errs)

def q2b(regr_t):
    """
    Q2 (b): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t'.
    """

    xs, errs = estimate_errors(regr_t, target_func)

    #plt.plot(xs, errs[0], 'r', label='Residual error')
    #plt.plot(xs, errs[1], 'g', label='Bias²')
    plt.plot(xs, errs[2], 'b', label='Variance')
    plt.plot(xs, errs[3], 'k', label='Expected error')

    plt.legend()
    plt.show()

def q2c(regr_t):
    """
    Q2 (c): Estimates the mean values of the the residual error, the squared
    bias, the variance and the expected error over the input space.
    """

    xs, errs = estimate_errors(regr_t, target_func)

    print(
        'Mean values over the input space ([{}; {}]) for {}:'
        '\tres. error: {:.3f}\tbias²: {:.3f}\tvar: {:.3f}\tsqu. err: {:.3f}'
        .format(
            RANGE_BEGIN, RANGE_END, regr_t.__name__,
            errs[0].mean(), errs[1].mean(), errs[2].mean(), errs[3].mean(),
        )
    )

def q2d(regr_t):
    """
    Q2 (d): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t' as a
    function of:
    * the size of the learning set;
    * the model complexity;
    * the standard deviation of the noise 'e';
    * the number of irrelevant input added to the problem, where these input
      are assumed to be independently drawn from U(-9, 9).
    """

    def estimate_errors_mean(ls_size):
        _, errs = estimate_errors(regr_t, target_func, ls_size=ls_size)
        return errs.mean(axis=1)

    # Tries several sizes of the learning set.
    ls_sizes = np.logspace(1, 4, 10)
    errs = np.empty((len(ls_sizes), 4))
    for i, ls_size in enumerate(ls_sizes):
        errs[i] = estimate_errors_mean(ls_size)

    plt.plot(ls_sizes, errs[:,0], 'r', label='Residual error')
    plt.plot(ls_sizes, errs[:,1], 'g', label='Bias²')
    plt.plot(ls_sizes, errs[:,2], 'b', label='Variance')
    plt.plot(ls_sizes, errs[:,3], 'k', label='Expected error')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    #q2b(LinearRegression)
    #q2b(DecisionTreeRegressor)

    q2c(LinearRegression)
    #q2c(DecisionTreeRegressor)
    q2c(KNeighborsRegressor)

    #q2d(LinearRegression)
    #q2d(KNeighborsRegressor)
