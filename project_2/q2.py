#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    N_SAMPLES = 10**6
    N_LS = 10000
    REGRESSION = LinearRegression

    assert N_SAMPLES % N_LS == 0

    np.random.seed(seed=1)

    #
    # Generates 'N_SAMPLES' samples.
    #

    xs = np.random.uniform(-9, 9, (N_SAMPLES, 1))

    es = np.random.normal(0, 1, (N_SAMPLES, 1))
    ys = (np.sin(xs) + np.cos(xs)) + xs * xs + es

    #
    # Computes the residual error by computing the standard deviation of the
    # sample set.
    #

    sigma2_ys = np.var(ys)

    print ("Residual error: {0:.3f}".format(sigma2_ys))

    #
    # Now we split the sample set in 'N_LS' learning sets. We train one
    # regression on each LS.
    #

    # We split the samples set by reshaping the numpy arrays. One line of the
    # new arrays define a LS.
    n_samples_ls = N_SAMPLES / N_LS                     # Samples per LS.
    xs_lss = np.reshape(xs, (N_LS, n_samples_ls, 1))
    ys_lss = np.reshape(ys, (N_LS, n_samples_ls, 1))

    # Trains one regression per LS.
    regrs = []
    for i in range(0, N_LS):
        xs_ls = xs_lss[i]
        ys_ls = ys_lss[i]

        regr = REGRESSION()
        regr.fit(xs_ls, ys_ls)

        regrs.append(regr)

    #
    # Computes the squared bias by computed the squared difference of the
    # mean of all the samples, and the mean of the average output of the trained
    # regressions on the samples.
    #

    mu_ys = np.mean(ys)

    # Computes E_xs { regr(xs) }, for each regression.
    mus_regrs = np.empty(len(regrs))
    for i, regr in enumerate(regrs):
        mus_regrs[i] = np.mean(regr.predict(xs))

    # Computes E_regrs { E_xs { regrs(xs) } }.
    mu_regrs = np.mean(mus_regrs)

    bias2 = mu_ys - mu_regrs
    print ("Bias²: {0:.3f}".format(bias2))

    var = np.mean((mus_regrs - mu_regrs)**2)
    print ("Variance: {0:.3f}".format(var))
