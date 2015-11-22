#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from estimate_error import estimate_errors_mean, plot_errors, target_func

# Question 2 (d): how many different values must be tried for each parameter.
Q2D_N_VALUES = 10

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
    q2d_ls(Ridge)
    q2d_ls(DecisionTreeRegressor)

    q2d_complexity(
        lambda alpha: lambda: Ridge(alpha=alpha), Ridge.__name__, 0, 5,
        x_label='alpha'
    )
    q2d_complexity(
        lambda max_depth: lambda: DecisionTreeRegressor(max_depth=max_depth),
        DecisionTreeRegressor.__name__, 1, 20, x_label='Max depth'
    )

    q2d_sigma(Ridge)
    q2d_sigma(DecisionTreeRegressor)

    q2d_noise(Ridge)
    q2d_noise(DecisionTreeRegressor)

