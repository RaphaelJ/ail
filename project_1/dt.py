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

from sklearn.tree import DecisionTreeClassifier

from data import make_data
from plot import plot_boundary

if __name__ == "__main__":
    # (Question 1) dt.py: Decision tree

    N_SAMPLES  = 2000
    N_TRAINING = 150

    X, y = make_data(N_SAMPLES, random_state=1)

    X_train, y_train = X[:N_TRAINING], y[:N_TRAINING]
    X_test,  y_test  = X[N_TRAINING:], y[N_TRAINING:]

    assert (len(X_train) == N_TRAINING)
    assert (len(y_train) == N_TRAINING)
    assert (len(X_test)  == N_SAMPLES - N_TRAINING)
    assert (len(y_test)  == N_SAMPLES - N_TRAINING)

    scores_test = []
    scores_train = []
    max_depth = 0
    depth = 1
    while True:
        dtc = DecisionTreeClassifier(max_depth=depth)
        dtc.fit(X_train, y_train)

        score_test = dtc.score(X_test, y_test)
        score_train = dtc.score(X_train, y_train)

        scores_test.append(score_test)
        scores_train.append(score_train)

        print("Depth: {} Score (test): {} Score (train): {}".format(
            depth, score_test, score_train
        ))

        plot_boundary("dt_out/dt_{depth}".format(depth=depth), dtc, X, y)

        # Stops when the tree perfectly fits the training set.
        if score_train == 1.0:
            break
        else:
            depth += 1
    plt.figure()
    plt.title("Error")
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.ylim(0, 1)
    plt.plot(range(1, len(scores_test) + 1), [1 - s for s in scores_test], 'r')
    plt.plot(range(1, len(scores_train) + 1), [1 - s for s in scores_train], 'g')
    plt.plot(9, 1-max(scores_test),"o")
    plt.savefig("dt_out/Scores.pdf")
    plt.close()

    

    
