#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from estimate_error import RANGE_BEGIN, RANGE_END, estimate_errors_mean, \
                           target_func

def q2c(regr_t):
    """
    Q2 (c): Estimates the mean values of the the residual error, the squared
    bias, the variance and the expected error over the input space.
    """

    means = estimate_errors_mean(regr_t, target_func)

    print(
        'Mean values over the input space ([{}; {}]) for {}:'
        '\tres. error: {:.3f}\tbias²: {:.3f}\tvar: {:.3f}\tsqu. err: {:.3f}'
        .format(
            RANGE_BEGIN, RANGE_END, regr_t.__name__, means[0], means[1],
            means[2], means[3],
        )
    )

if __name__ == "__main__":
    q2c(Ridge)
    q2c(DecisionTreeRegressor)
    q2c(KNeighborsRegressor)
