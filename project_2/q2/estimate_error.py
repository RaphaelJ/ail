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
from sklearn.metrics import mean_squared_error

N_LS = 100
LS_SIZE = 10**3

RANGE_BEGIN = -9
RANGE_END = 9
RANGE_SIZE = 200

N_SAMPLES_ESTIMATE = 10**5

def target_func(xs, sigma=1.0, noise=0.0):
    """
    Given a matrix of samples and their single attribute 'x', returns a matrix
    of values computed as '(sin(x) + cos(x)) * x² + e' where 'e ~ N(0, sigma)'
    is a noise variable.

    Parameters
    ----------
    xs : array-like, shape = [n_samples, 1]
         The input samples
    sigma : the standard deviation of the noise 'e'.
    noise : the probablity (between 0 and 1) of the function to generate an
            irrelevant noisy output for a given input sample (noise=0 : only
            regular outputs, noise=1 : only irrelevant outputs).

    Returns
    -------
    ys : array-like, shape = [n_samples, 1]
         The target values.
    """

    assert 0.0 <= noise <= 1.0

    n_samples = xs.shape[0]
    xs = xs.reshape(n_samples) # Works with vectors instead of matrices.

    # Computes the output of the function for all 'xs'.
    if sigma > 0:
        es = np.random.normal(0, sigma, n_samples)
    else:
        es = np.zeros(n_samples)
    regular_ys = (np.sin(xs) + np.cos(xs)) * xs * xs + es

    # Computes noisy values for all 'xs'.
    noisy_ys = np.random.uniform(-9, 9, n_samples)

    # Creates a random vector of 0/1 values. Each value indicates for a sample
    # in 'xs' if an irrelevant/noisy output must be returned (1) or not (0).
    is_noise = np.random.choice([0, 1], n_samples, p=[1.0 - noise, noise])

    # Selects either the noisy value, or the real output, depending of
    # 'is_noise'
    ys = regular_ys * np.logical_not(is_noise) + noisy_ys * is_noise

    return ys.reshape((n_samples, 1))

def plot_target_func(sigma=0.0, noise=0.0):
    """
    Plots '(sin(x) + cos(x)) * x² + e' where 'e ~ N(0, sigma)' is a noise
    variable.
    """

    xs = np.linspace(RANGE_BEGIN, RANGE_END, RANGE_SIZE)

    plt.plot(xs, target_func(xs, sigma=sigma, noise=noise), 'r')
    plt.title('sin(x) + cos(x)) * x²')
    plt.show()

def estimate_error(f, regrs, x):
    """
    Estimates the residual error, the squared bias, the variance and the
    expected mean square error of the given regressions 'regrs', for a function
    'f()' at the given point 'x'.

    Does this by generating a large number of samples ('N_SAMPLES_ESTIMATE') for
    the value 'x' using 'f()'.

    Parameters
    ----------
    f : the function to evaluate. The function must accept a matrix of samples.
    regrs : a list a regressions that have been trained on the function.
    x : the value of 'x' for which the function error must be evaluated.

    Returns
    -------
    (res_error, sq_bias, var, mse) : a four tuple containing the estimated
                                     errors.
    """

    # Generates 'N_SAMPLES_ESTIMATE' samples by running the function 'f()' on
    # point 'x'.

    xs = np.empty((N_SAMPLES_ESTIMATE, 1))
    xs.fill(x)
    ys = f(xs)

    #
    # Computes the residual error by computing the variance of the sample set
    # generated on point 'x'.
    #

    res_error = np.var(ys)

    #
    # Computes the squared bias by computing the mean squared difference of the
    # mean of all the samples, and the mean of the predicted outputs of the
    # trained regressions on 'x'.
    #

    mu_ys = np.mean(ys)

    ys_regrs = np.array([regr.predict(x)[0] for regr in regrs])

    mu_ys_regrs = np.mean(ys_regrs)

    sq_bias = np.mean((mu_ys - mu_ys_regrs)**2)

    # The variance is just the mean squared difference between the predicted
    # outputs and the mean of the predicted outputs.

    var = np.mean((ys_regrs - mu_ys_regrs)**2)

    # Mean square error (MSE).
    mse = res_error + sq_bias + var

    # Alternative way of computing the error rate, by testing the regressions
    # using the 'mean_squared_error()' function:
    # xs2 = np.empty((N_LS, 1))
    # xs2.fill(x)
    # mse = mean_squared_error(f(xs2), ys_regrs)

    return (res_error, sq_bias, var, mse)

def estimate_errors(regr_t, f, ls_size=LS_SIZE):
    """
    Trains 'n_ls' regressions on 'N_LS' randomly generated training set of size
    'ls_size' and estimate their errors with 'estimate_error()' on a range of
    'x' values determined by 'RANGE_BEGIN', 'RANGE_END' and 'RANGE_SIZE'.

    Parameters
    ----------
    regr_t : a function that creates a new regression object.
    f : the function to evaluate. The function must accept a matrix of samples.
    ls_size : the number of samples used to train a single regression.

    Returns
    -------
    xs, errs where:
        xs is an Numpy vector of length 'RANGE_SIZE' containing the 'x' values
            between 'RANGE_BEGIN' and 'RANGE_END' on which the errors have been
            evaluated
        errs is a Numpy matrix of shape (RANGE_SIZE, 4) containing the four
            error values (residual error, bias², variance and squared error) for
            each 'x' value in 'xs'.
    """

    def train_regression():
        xs_train = np.random.uniform(RANGE_BEGIN, RANGE_END, (ls_size, 1))
        regr = regr_t()
        regr.fit(xs_train, f(xs_train))
        return regr

    regrs = [train_regression() for i in range(0, N_LS)]

    xs = np.linspace(RANGE_BEGIN, RANGE_END, RANGE_SIZE)
    errs = np.vectorize(lambda x: estimate_error(f, regrs, x))(xs)

    return xs, np.array(errs).T

def estimate_errors_mean(regr_t, f, ls_size=LS_SIZE):
    """
    Returns the means values of the errors values estimated with
    'estimate_errors()' over the whole function space
    ('[RANGE_BEGIN, RANGE_END]').

    Parameters
    ----------
    regr_t : a function that creates a new regression object.
    f : the function to evaluate. The function must accept a matrix of samples.
    ls_size : the number of samples used to train a single regression.

    Returns
    -------
    errs : a Numpy vector of length 4, containing the four mean error values
           (residual error, bias², variance and squared error).
    """

    _, errs = estimate_errors(regr_t, f, ls_size=ls_size)
    return errs.mean(axis=0)

def plot_errors(xs, errs, title=None, residual=True, x_label=None):
    if residual:
        plt.plot(xs, errs[:,0], 'k', label='Residual error')

    plt.plot(xs, errs[:,1], 'g', label='Bias²')
    plt.plot(xs, errs[:,2], 'b', label='Variance')
    plt.plot(xs, errs[:,3], 'r', label='Expected error')

    plt.legend()

    if x_label is not None:
        plt.xlabel(x_label)

    if title is not None:
        plt.title(title)

    plt.show()

def q2d_ls(regr_t):
    """
    Q2 (d): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t' as a
    function of the size of the learning set
    """

    ls_sizes = np.logspace(1, 2.5, 10)
    errs = np.array([
        estimate_errors_mean(regr_t, target_func, ls_size=ls_size)
        for ls_size in ls_sizes
    ])
    plot_errors(
        ls_sizes, errs,
        title='Mean error values as a function of the LS size for {}'.format(
            regr_t.__name__
        ), residual=False, x_label='Samples in LS'
    )

def q2d_complexity(regr_t, regr_t_name, min_cplx, max_cplx, x_label):
    """
    Q2 (d): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t' as a
    function of the model complexity.

    The 'regr_t' function creates a new regression object constructor (e.g. a
    function) from a complexity value in the range between 'min_cplx' and
    'max_cplx'.
    """

    cplxs = np.linspace(min_cplx, max_cplx, 10)
    errs = np.array([
        estimate_errors_mean(regr_t(cplx), target_func) for cplx in cplxs
    ])
    plot_errors(
        cplxs, errs,
        title='Mean error values as a function of the complexity of {}'.format(
            regr_t_name
        ), residual=False, x_label=x_label
    )


def q2d_sigma(regr_t):
    """
    Q2 (d): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t' as a
    function of the standard deviation of the noise 'e'.
    """

    sigmas = np.linspace(0, 20, 10)
    errs = np.array([
        estimate_errors_mean(regr_t, lambda xs: target_func(xs, sigma=sigma))
        for sigma in sigmas
    ])
    plot_errors(
        sigmas, errs,
        title='Mean error values as a function of the noise for {}'.format(
            regr_t.__name__
        ), x_label='Standard deviation of e'
    )

def q2d_noise(regr_t):
    """
    Q2 (d): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t' as a
    function of the number of irrelevant input added to the problem, where these
    input are assumed to be independently drawn from U(-9, 9).
    """

    noises = np.linspace(0.0, 0.5, 10)
    errs = np.array([
        estimate_errors_mean(regr_t, lambda xs: target_func(xs, noise=noise))
        for noise in noises
    ])
    plot_errors(
        noises, errs,
        title='Mean error values as a function of noise for {}'.format(
            regr_t.__name__
        ), x_label='Irrelevant input ratio'
    )

if __name__ == "__main__":
    #plot_target_func()

    #q2b(Ridge)
    #q2b(DecisionTreeRegressor)

    #q2c(Ridge)
    #q2c(DecisionTreeRegressor)
    #q2c(KNeighborsRegressor)

    #q2d_ls(Ridge)
    #q2d_ls(DecisionTreeRegressor)

    #q2d_complexity(
        #lambda alpha: lambda: Ridge(alpha=alpha), Ridge.__name__, 0, 5,
        #x_label='alpha'
    #)
    #q2d_complexity(
        #lambda max_depth: lambda: DecisionTreeRegressor(max_depth=max_depth),
        #DecisionTreeRegressor.__name__, 1, 20, x_label='Max depth'
    #)

    #q2d_sigma(Ridge)
    #q2d_sigma(DecisionTreeRegressor)

    q2d_noise(Ridge)
    q2d_noise(DecisionTreeRegressor)

