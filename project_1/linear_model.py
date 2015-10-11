#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifier

from data import make_data
from plot import plot_boundary


if __name__ == "__main__":
    # (Question 3) ols.py: Ordinary least square
    # TODO your code here
