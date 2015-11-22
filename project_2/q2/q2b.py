#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from estimate_error import estimate_errors, plot_errors, target_func

def q2b(regr_t):
    """
    Q2 (b): Plots the residual error, the squared bias, the variance and the
    expected error as a function of x for the regression method 'regr_t'.
    """

    xs, errs = estimate_errors(regr_t, target_func)
    plot_errors(
        xs, errs, title='Error decomposition for {}'.format(regr_t.__name__)
    )

if __name__ == "__main__":
    q2b(Ridge)
    q2b(DecisionTreeRegressor)
