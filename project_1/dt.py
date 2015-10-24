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

from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from data import make_data
from plot import plot_boundary

if __name__ == "__main__":
    # (Question 1) dt.py: Decision tree

    N_SAMPLES  = 2000
    N_TRAINING = 150
    RANDOM_STATE = 1

    MAX_DEPTH = 50 # Doesn't train tree deeper than 'MAX_DEPTH'

    X, y = make_data(N_SAMPLES, random_state=RANDOM_STATE)

    # Splits the training and testings sets randomly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_TRAINING, random_state=RANDOM_STATE
    )

    assert (len(X_train) == N_TRAINING)
    assert (len(y_train) == N_TRAINING)
    assert (len(X_test)  == N_SAMPLES - N_TRAINING)
    assert (len(y_test)  == N_SAMPLES - N_TRAINING)

    scores_test = []
    scores_train = []
    scores_ten_fold = []

    for depth in range(1, MAX_DEPTH + 1):
        dtc = DecisionTreeClassifier(max_depth=depth)
        # Case with random state inside of the tree classifier
        """dtc = DecisionTreeClassifier(max_depth=depth,
                                     random_state = RANDOM_STATE)"""

        # Trains the classifier on a simple train/test split of the data.
        dtc.fit(X_train, y_train)

        score_test = dtc.score(X_test, y_test)
        score_train = dtc.score(X_train, y_train)

        scores_test.append(score_test)
        scores_train.append(score_train)

        # Computes the score using a 10-fold cross-validation.
        score_ten_fold = cross_val_score(dtc, X, y, cv=10).mean()
        scores_ten_fold.append(score_ten_fold)

        print(
            "Depth: {} Score (test): {} Score (train): {} Score (10-fold): {}"
                .format(depth, score_test, score_train, score_ten_fold)
        )

        plot_boundary(
            "dt_out/dt_{depth}".format(depth=depth), dtc, X_test, y_test
        )

    plt.figure()
    plt.title("Decision tree error rate")
    plt.xlabel("Tree depth (complexity)")
    plt.ylabel("Error")
    plt.ylim(0, 0.6)
    depth_range = range(1, len(scores_test) + 1)
    plt.plot(
        depth_range, [1 - s for s in scores_test], 'r', label="Testing set"
    )
    plt.plot(
        depth_range, [1 - s for s in scores_train], 'g', label="Training set"
    )
    plt.plot(
        depth_range, [1 - s for s in scores_ten_fold], 'b',
        label="10-fold cross-validation"
    )

    def find_best_score(scores):
        """
        Given a list of scores, returns the best score, and the depth at which
        it was achieved.
        """
        best_score = max(scores)
        best_score_depth = scores.index(best_score) + 1
        return best_score, best_score_depth

    # Find best scores
    best_test_score, best_test_depth = find_best_score(scores_test)
    plt.plot(best_test_depth, 1 - best_test_score, "ro")

    best_ten_fold_score, best_ten_fold_depth = find_best_score(scores_ten_fold)
    plt.plot(best_ten_fold_depth, 1 - best_ten_fold_score, "bo")

    plt.legend()

    plt.savefig("dt_out/Scores.pdf")
    plt.close()
